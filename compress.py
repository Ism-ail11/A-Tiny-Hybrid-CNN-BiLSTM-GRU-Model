from __future__ import annotations
import os, argparse, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from src.utils.config import load_yaml
from src.utils.paths import ensure_dir
from src.data.daisee_dataset import DAiSEEClipDataset
from src.data.splits import subject_split
from src.models.tiny_hybrid import TinyHybridAttentionModel, BranchEncoder, FrameCNN
from src.models.temperature import fit_temperature

CLASSES = ['Very Low','Low','High','Very High']

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='runs/compress')
    ap.add_argument('--export_name', default='model')
    return ap.parse_args()

def device_from_cfg(cfg):
    if cfg.get('device','auto') == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg['device']

# --------- Structured pruning helpers (rebuild modules) ---------
def _topk_indices_by_l1(weight, keep_ratio):
    # weight: (out_ch, ...)
    out_ch = weight.shape[0]
    keep = max(1, int(round(out_ch * keep_ratio)))
    scores = weight.abs().reshape(out_ch, -1).sum(dim=1)
    idx = torch.argsort(scores, descending=True)[:keep]
    idx_sorted, _ = torch.sort(idx)
    return idx_sorted

def prune_frame_cnn(cnn: FrameCNN, keep_ratio: float):
    # Prune conv1 out channels then conv2 out channels; adjust in channels accordingly.
    device = next(cnn.parameters()).device
    with torch.no_grad():
        idx1 = _topk_indices_by_l1(cnn.conv1.weight.data, keep_ratio)
        # rebuild conv1/bn1
        conv1 = nn.Conv2d(cnn.conv1.in_channels, len(idx1), 3, 1, 1, bias=False).to(device)
        bn1 = nn.BatchNorm2d(len(idx1)).to(device)
        conv1.weight.copy_(cnn.conv1.weight.data[idx1])
        bn1.weight.copy_(cnn.bn1.weight.data[idx1])
        bn1.bias.copy_(cnn.bn1.bias.data[idx1])
        bn1.running_mean.copy_(cnn.bn1.running_mean.data[idx1])
        bn1.running_var.copy_(cnn.bn1.running_var.data[idx1])

        # conv2 depends on conv1 channels
        w2 = cnn.conv2.weight.data[:, idx1, :, :]
        idx2 = _topk_indices_by_l1(w2, keep_ratio)
        conv2 = nn.Conv2d(len(idx1), len(idx2), 3, 1, 1, bias=False).to(device)
        bn2 = nn.BatchNorm2d(len(idx2)).to(device)
        conv2.weight.copy_(w2[idx2])
        bn2.weight.copy_(cnn.bn2.weight.data[idx2])
        bn2.bias.copy_(cnn.bn2.bias.data[idx2])
        bn2.running_mean.copy_(cnn.bn2.running_mean.data[idx2])
        bn2.running_var.copy_(cnn.bn2.running_var.data[idx2])

        # proj linear: in_features == conv2 out channels after GAP
        old_lin = cnn.proj[1]
        new_lin = nn.Linear(len(idx2), old_lin.out_features).to(device)
        # old weight: (out_features, in_features_old)
        new_lin.weight.copy_(old_lin.weight.data[:, idx2])
        new_lin.bias.copy_(old_lin.bias.data)

        # swap in
        cnn.conv1, cnn.bn1 = conv1, bn1
        cnn.conv2, cnn.bn2 = conv2, bn2
        cnn.proj[1] = new_lin

def _select_gate_rows(indices, gate_mult):
    # For LSTM: gate_mult=4; GRU: gate_mult=3
    rows = []
    h = len(indices)
    # indices are in [0, hidden_old)
    for g in range(gate_mult):
        rows.append(indices + g * h)
    return torch.cat(rows, dim=0)

def prune_lstm_module(lstm: nn.LSTM, keep_ratio: float):
    """Prunes hidden units of a 1-layer LSTM (possibly bidirectional) by rebuilding module."""
    device = next(lstm.parameters()).device
    input_size = lstm.input_size
    hidden_size = lstm.hidden_size
    num_layers = lstm.num_layers
    assert num_layers == 1, "This helper supports num_layers=1"
    bidir = lstm.bidirectional
    keep = max(1, int(round(hidden_size * keep_ratio)))

    def prune_direction(suffix):
        w_ih = getattr(lstm, f'weight_ih_l0{suffix}').data  # (4h, input)
        w_hh = getattr(lstm, f'weight_hh_l0{suffix}').data  # (4h, h)
        b_ih = getattr(lstm, f'bias_ih_l0{suffix}').data
        b_hh = getattr(lstm, f'bias_hh_l0{suffix}').data

        # importance per unit: l2 norm over recurrent rows (sum gates)
        # compute score per unit j across gates rows
        scores = []
        for j in range(hidden_size):
            rows = torch.tensor([j, j+hidden_size, j+2*hidden_size, j+3*hidden_size], device=device)
            scores.append(w_hh[rows].abs().sum().item() + w_ih[rows].abs().sum().item())
        scores = torch.tensor(scores, device=device)
        idx = torch.argsort(scores, descending=True)[:keep]
        idx, _ = torch.sort(idx)

        # build pruned weights
        # gate rows
        gate_rows = torch.cat([idx + g*hidden_size for g in range(4)], dim=0)
        new_w_ih = w_ih[gate_rows, :].clone()
        new_w_hh = w_hh[gate_rows, :][:, idx].clone()
        new_b_ih = b_ih[gate_rows].clone()
        new_b_hh = b_hh[gate_rows].clone()
        return idx, new_w_ih, new_w_hh, new_b_ih, new_b_hh

    # prune forward
    idx_f, wih_f, whh_f, bih_f, bhh_f = prune_direction('')
    if bidir:
        idx_b, wih_b, whh_b, bih_b, bhh_b = prune_direction('_reverse')

    new_hidden = len(idx_f)  # keep same for both directions; use forward keep
    new_lstm = nn.LSTM(input_size=input_size, hidden_size=new_hidden, num_layers=1,
                       batch_first=True, bidirectional=bidir, dropout=0.0).to(device)

    # copy weights
    with torch.no_grad():
        new_lstm.weight_ih_l0.copy_(wih_f)
        new_lstm.weight_hh_l0.copy_(whh_f)
        new_lstm.bias_ih_l0.copy_(bih_f)
        new_lstm.bias_hh_l0.copy_(bhh_f)
        if bidir:
            new_lstm.weight_ih_l0_reverse.copy_(wih_b)
            new_lstm.weight_hh_l0_reverse.copy_(whh_b)
            new_lstm.bias_ih_l0_reverse.copy_(bih_b)
            new_lstm.bias_hh_l0_reverse.copy_(bhh_b)

    return new_lstm

def prune_gru_module(gru: nn.GRU, keep_ratio: float):
    """Prunes hidden units of a 1-layer GRU by rebuilding module."""
    device = next(gru.parameters()).device
    input_size = gru.input_size
    hidden_size = gru.hidden_size
    num_layers = gru.num_layers
    assert num_layers == 1 and not gru.bidirectional, "This helper supports 1-layer unidirectional GRU"
    keep = max(1, int(round(hidden_size * keep_ratio)))

    w_ih = gru.weight_ih_l0.data  # (3h, input)
    w_hh = gru.weight_hh_l0.data  # (3h, h)
    b_ih = gru.bias_ih_l0.data
    b_hh = gru.bias_hh_l0.data

    # score per unit across gates
    scores = []
    for j in range(hidden_size):
        rows = torch.tensor([j, j+hidden_size, j+2*hidden_size], device=device)
        scores.append(w_hh[rows].abs().sum().item() + w_ih[rows].abs().sum().item())
    scores = torch.tensor(scores, device=device)
    idx = torch.argsort(scores, descending=True)[:keep]
    idx, _ = torch.sort(idx)

    gate_rows = torch.cat([idx + g*hidden_size for g in range(3)], dim=0)
    new_w_ih = w_ih[gate_rows, :].clone()
    new_w_hh = w_hh[gate_rows, :][:, idx].clone()
    new_b_ih = b_ih[gate_rows].clone()
    new_b_hh = b_hh[gate_rows].clone()

    new_hidden = len(idx)
    new_gru = nn.GRU(input_size=input_size, hidden_size=new_hidden, num_layers=1,
                     batch_first=True, bidirectional=False, dropout=0.0).to(device)
    with torch.no_grad():
        new_gru.weight_ih_l0.copy_(new_w_ih)
        new_gru.weight_hh_l0.copy_(new_w_hh)
        new_gru.bias_ih_l0.copy_(new_b_ih)
        new_gru.bias_hh_l0.copy_(new_b_hh)
    return new_gru

def structured_prune_model(model: TinyHybridAttentionModel, conv_keep: float, rnn_keep: float):
    """Applies structured pruning by rebuilding CNN/RNN modules inside each branch."""
    for b in model.branches:
        # CNN
        prune_frame_cnn(b.cnn, conv_keep)
        # BiLSTM, GRU, fusion LSTM
        b.bilstm = prune_lstm_module(b.bilstm, rnn_keep)
        # GRU input size changes because BiLSTM hidden size changed
        # Update GRU input_size to match 2*hidden
        new_in = 2 * b.bilstm.hidden_size
        # rebuild GRU with same hidden before pruning? We prune old one directly but need to adapt input projection
        # easiest: create a fresh GRU matching new input size, then copy overlapping weights where possible
        old_gru = b.gru
        # create compatible GRU then prune it
        tmp_gru = nn.GRU(input_size=new_in, hidden_size=old_gru.hidden_size, num_layers=1, batch_first=True)
        tmp_gru = tmp_gru.to(next(old_gru.parameters()).device)
        b.gru = tmp_gru
        b.gru = prune_gru_module(b.gru, rnn_keep)

        # fusion lstm input size becomes new gru hidden
        old_fus = b.fusion_lstm
        tmp_lstm = nn.LSTM(input_size=b.gru.hidden_size, hidden_size=old_fus.hidden_size, num_layers=1, batch_first=True)
        tmp_lstm = tmp_lstm.to(next(old_fus.parameters()).device)
        b.fusion_lstm = tmp_lstm
        b.fusion_lstm = prune_lstm_module(b.fusion_lstm, rnn_keep)

        b.out_dim = b.fusion_lstm.hidden_size

    # head in_features changes
    fused_dim = sum(b.out_dim for b in model.branches)
    # rebuild head with same layout but new dims
    old_head = model.head
    new_head = nn.Sequential(
        nn.Linear(fused_dim, fused_dim//2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(fused_dim//2, old_head[-1].out_features)
    ).to(next(model.parameters()).device)
    model.head = new_head

# --------- Export / quant / temp scaling ---------
def export_onnx(model, out_path, num_frames=30, img_size=224, device='cpu'):
    model.eval()
    dummy = torch.randn(1, num_frames, 3, img_size, img_size, device=device)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=['clip'], output_names=['logits'],
        dynamic_axes={'clip': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=17
    )
    onnx.checker.check_model(onnx.load(out_path))

def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    ensure_dir(args.out)
    device = device_from_cfg(cfg)

    # data split to build val loader for temp scaling
    root = cfg['data']['root']
    videos_dir = cfg['data']['videos_dir']
    ann_dir = cfg['data']['annotations_dir']
    tmp = DAiSEEClipDataset(root, videos_dir, ann_dir, ids=[], num_frames=cfg['data']['num_frames'])
    all_ids = sorted(list(tmp.labels.keys()))
    train_ids, val_ids, test_ids, meta = subject_split(
        all_ids, cfg['data']['split']['train'], cfg['data']['split']['val'], cfg['data']['split']['test'],
        seed=int(cfg['seed'])
    )

    ds_val = DAiSEEClipDataset(
        root, videos_dir, ann_dir, val_ids,
        num_frames=cfg['data']['num_frames'],
        img_size=cfg['data']['img_size'],
        brightness_target=cfg['data']['brightness_target'],
        brightness_strength=cfg['data']['brightness_strength']
    )
    dl_val = DataLoader(ds_val, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=cfg['train']['num_workers'])

    # load model
    branch_cfgs = cfg['model']['branches']
    model = TinyHybridAttentionModel(cfg['model']['num_classes'], cfg['model']['embedding_dim'], branch_cfgs)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(device)

    # structured prune
    if cfg['compression']['prune']['enabled']:
        conv_keep = float(cfg['compression']['prune']['conv_keep_ratio'])
        rnn_keep = float(cfg['compression']['prune']['rnn_keep_ratio'])
        structured_prune_model(model, conv_keep=conv_keep, rnn_keep=rnn_keep)

    # temperature scaling
    scaler = None
    if cfg['compression']['temp_scaling']['enabled']:
        scaler = fit_temperature(model, dl_val, device=device)
        # wrap: we will export scaled logits by applying scaler before softmax at inference time
        class Wrapped(nn.Module):
            def __init__(self, core, scaler):
                super().__init__()
                self.core = core
                self.scaler = scaler
            def forward(self, x):
                return self.scaler(self.core(x))
        model_to_export = Wrapped(model, scaler).to(device)
    else:
        model_to_export = model

    # export ONNX (fp32 or pruned fp32)
    onnx_fp = os.path.join(args.out, f"{args.export_name}_fp32.onnx")
    export_onnx(model_to_export, onnx_fp, num_frames=cfg['data']['num_frames'], img_size=cfg['data']['img_size'], device=device)

    # quantize
    onnx_int8 = None
    if cfg['compression']['quant']['enabled']:
        onnx_int8 = os.path.join(args.out, f"{args.export_name}_int8.onnx")
        # Dynamic quantization: works well on CPU and is the most robust option across ops.
        quantize_dynamic(
            model_input=onnx_fp,
            model_output=onnx_int8,
            weight_type=QuantType.QInt8
        )

    # save compression meta
    meta_out = {
        'onnx_fp32': onnx_fp,
        'onnx_int8': onnx_int8,
        'temperature': (scaler.temperature if scaler is not None else None)
    }
    with open(os.path.join(args.out, 'compression_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta_out, f, indent=2)

    print(json.dumps(meta_out, indent=2))

if __name__ == '__main__':
    main()
