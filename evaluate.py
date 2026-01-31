from __future__ import annotations
import os, argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef, classification_report
)
from scipy.stats import spearmanr

from src.utils.config import load_yaml
from src.utils.paths import ensure_dir
from src.data.daisee_dataset import DAiSEEClipDataset
from src.data.splits import subject_split
from src.models.tiny_hybrid import TinyHybridAttentionModel
from src.utils.metrics import ece, brier_multi, ordinal_mae, accuracy
from src.utils.plotting import plot_confusion, plot_reliability

CLASSES = ['Very Low','Low','High','Very High']

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='runs/eval')
    ap.add_argument('--split', choices=['train','val','test'], default='test')
    return ap.parse_args()

def device_from_cfg(cfg):
    if cfg.get('device','auto') == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg['device']

def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    ensure_dir(args.out)
    device = device_from_cfg(cfg)

    root = cfg['data']['root']
    videos_dir = cfg['data']['videos_dir']
    ann_dir = cfg['data']['annotations_dir']

    tmp = DAiSEEClipDataset(root, videos_dir, ann_dir, ids=[], num_frames=cfg['data']['num_frames'])
    all_ids = sorted(list(tmp.labels.keys()))
    train_ids, val_ids, test_ids, meta = subject_split(
        all_ids, cfg['data']['split']['train'], cfg['data']['split']['val'], cfg['data']['split']['test'],
        seed=int(cfg['seed'])
    )
    ids = {'train': train_ids, 'val': val_ids, 'test': test_ids}[args.split]

    ds = DAiSEEClipDataset(
        root, videos_dir, ann_dir, ids,
        num_frames=cfg['data']['num_frames'],
        img_size=cfg['data']['img_size'],
        brightness_target=cfg['data']['brightness_target'],
        brightness_strength=cfg['data']['brightness_strength']
    )
    dl = DataLoader(ds, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=cfg['train']['num_workers'])

    branch_cfgs = cfg['model']['branches']
    model = TinyHybridAttentionModel(cfg['model']['num_classes'], cfg['model']['embedding_dim'], branch_cfgs)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(device).eval()

    y_true, y_pred, probs = [], [], []
    with torch.no_grad():
        for x, y, _ in tqdm(dl, desc="Eval"):
            x = x.to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1)
            pred = p.argmax(dim=1)
            y_true.extend(y.numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            probs.append(p.cpu().numpy())
    probs = np.concatenate(probs, axis=0)

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)

    # Aggregate metrics
    out = {}
    out['accuracy'] = accuracy(y_true_np, y_pred_np)
    out['macro_f1'] = float(f1_score(y_true_np, y_pred_np, average='macro'))
    out['macro_precision'] = float(precision_score(y_true_np, y_pred_np, average='macro', zero_division=0))
    out['macro_recall'] = float(recall_score(y_true_np, y_pred_np, average='macro', zero_division=0))
    out['qwk'] = float(cohen_kappa_score(y_true_np, y_pred_np, weights='quadratic'))
    out['ordinal_mae'] = ordinal_mae(y_true_np, y_pred_np)
    out['spearman_r'] = float(spearmanr(y_true_np, y_pred_np).correlation)
    out['mcc'] = float(matthews_corrcoef(y_true_np, y_pred_np))
    # AUC
    try:
        out['macro_auc_ovr'] = float(roc_auc_score(y_true_np, probs, multi_class='ovr', average='macro'))
    except Exception as e:
        out['macro_auc_ovr'] = None
        out['auc_error'] = str(e)

    out['brier'] = brier_multi(probs, y_true_np, num_classes=cfg['model']['num_classes'])
    out['ece'] = ece(probs, y_true_np, n_bins=15)

    # Save report
    report = classification_report(y_true_np, y_pred_np, target_names=CLASSES, digits=4, zero_division=0)
    with open(os.path.join(args.out, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

    with open(os.path.join(args.out, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    # plots
    plot_confusion(y_true_np, y_pred_np, CLASSES, os.path.join(args.out, 'confusion.png'))
    plot_reliability(probs, y_true_np, os.path.join(args.out, 'reliability.png'))

    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
