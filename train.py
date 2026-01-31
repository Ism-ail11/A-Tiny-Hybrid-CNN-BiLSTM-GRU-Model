from __future__ import annotations
import os, argparse, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from src.utils.config import load_yaml
from src.utils.seed import set_seed, set_deterministic
from src.utils.paths import ensure_dir
from src.data.daisee_dataset import DAiSEEClipDataset
from src.data.splits import subject_split
from src.models.tiny_hybrid import TinyHybridAttentionModel
from src.train_utils import make_warmup_cosine_scheduler, class_weights_from_counts

CLASSES = ['Very Low','Low','High','Very High']

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--out', default='runs/exp1')
    ap.add_argument('--no_deterministic', action='store_true')
    return ap.parse_args()

def device_from_cfg(cfg):
    if cfg.get('device','auto') == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg['device']

def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    ensure_dir(args.out)

    set_seed(int(cfg['seed']))
    if not args.no_deterministic:
        set_deterministic()

    device = device_from_cfg(cfg)

    # Build ID list from annotation keys (robust to missing files)
    root = cfg['data']['root']
    videos_dir = cfg['data']['videos_dir']
    ann_dir = cfg['data']['annotations_dir']
    # We load labels from dataset class to reuse parsing rules
    tmp = DAiSEEClipDataset(root, videos_dir, ann_dir, ids=[], num_frames=cfg['data']['num_frames'])
    all_ids = sorted(list(tmp.labels.keys()))
    train_ids, val_ids, test_ids, meta = subject_split(
        all_ids, cfg['data']['split']['train'], cfg['data']['split']['val'], cfg['data']['split']['test'],
        seed=int(cfg['seed'])
    )
    with open(os.path.join(args.out, 'split_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    ds_train = DAiSEEClipDataset(
        root, videos_dir, ann_dir, train_ids,
        num_frames=cfg['data']['num_frames'],
        img_size=cfg['data']['img_size'],
        brightness_target=cfg['data']['brightness_target'],
        brightness_strength=cfg['data']['brightness_strength']
    )
    ds_val = DAiSEEClipDataset(
        root, videos_dir, ann_dir, val_ids,
        num_frames=cfg['data']['num_frames'],
        img_size=cfg['data']['img_size'],
        brightness_target=cfg['data']['brightness_target'],
        brightness_strength=cfg['data']['brightness_strength']
    )

    dl_train = DataLoader(ds_train, batch_size=cfg['train']['batch_size'], shuffle=True,
                          num_workers=cfg['train']['num_workers'], pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=cfg['train']['batch_size'], shuffle=False,
                        num_workers=cfg['train']['num_workers'], pin_memory=True)

    # class weights
    counts = [0,0,0,0]
    for _, y, _ in dl_train:
        for yi in y.tolist():
            counts[int(yi)] += 1
    w = class_weights_from_counts(counts).to(device)

    branch_cfgs = cfg['model']['branches']
    model = TinyHybridAttentionModel(
        num_classes=cfg['model']['num_classes'],
        emb_dim=cfg['model']['embedding_dim'],
        branch_cfgs=branch_cfgs
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    scheduler = make_warmup_cosine_scheduler(opt, cfg['train']['warmup_epochs'], cfg['train']['epochs'], len(dl_train))

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg['train']['amp']) and device=='cuda')
    ce = nn.CrossEntropyLoss(weight=w)

    best_f1 = -1.0
    patience = 0
    best_path = os.path.join(args.out, 'best.pt')

    global_step = 0
    for epoch in range(1, cfg['train']['epochs'] + 1):
        model.train()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{cfg['train']['epochs']}")
        for x, y, _ in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(x)
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['train']['grad_clip_norm'])
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            global_step += 1
            pbar.set_postfix(loss=float(loss.item()), lr=float(opt.param_groups[0]['lr']))

        # validation
        model.eval()
        y_true, y_pred = [], []
        val_loss = 0.0
        with torch.no_grad():
            for x, y, _ in dl_val:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = ce(logits, y)
                val_loss += float(loss.item()) * x.size(0)
                pred = logits.argmax(dim=1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(pred.cpu().tolist())
        val_loss /= max(1, len(ds_val))
        macro_f1 = float(f1_score(y_true, y_pred, average='macro'))
        print(f"[VAL] epoch={epoch} loss={val_loss:.4f} macro_f1={macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            patience = 0
            torch.save({'model': model.state_dict(), 'cfg': cfg}, best_path)
            print(f"Saved best -> {best_path}")
        else:
            patience += 1
            if patience >= cfg['train']['early_stop_patience']:
                print("Early stopping.")
                break

    print(f"Done. Best macro-F1={best_f1:.4f}")

if __name__ == '__main__':
    main()
