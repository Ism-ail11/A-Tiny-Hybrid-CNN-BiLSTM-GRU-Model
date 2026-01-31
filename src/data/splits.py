from __future__ import annotations
import os, json, re
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

SUBJECT_RE = re.compile(r"(subject|sub|s)(\d+)", re.IGNORECASE)

def infer_subject_id(video_id: str) -> str:
    """Best-effort subject id extraction from DAiSEE-style IDs.
    If your IDs differ, override by providing a mapping file.
    """
    m = SUBJECT_RE.search(video_id)
    if m:
        return m.group(2)
    # fallback: prefix before first underscore
    return video_id.split('_')[0]

def load_attention_labels(annotations_dir: str) -> Dict[str, int]:
    """Loads attention labels from a JSON file structure.
    Expected: a JSON mapping {video_id: label_string_or_int}.
    If you have multiple JSONs, we merge them (later files override earlier).
    label_string must be one of: Very Low, Low, High, Very High (case-insensitive).
    """
    label_map = {}
    if not os.path.isdir(annotations_dir):
        raise FileNotFoundError(f"annotations_dir not found: {annotations_dir}")

    json_files = [f for f in os.listdir(annotations_dir) if f.lower().endswith('.json')]
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {annotations_dir}")

    def to_int(v):
        if isinstance(v, int):
            return v
        s = str(v).strip().lower()
        if s in ['very low', 'very_low', 'verylow', '0']:
            return 0
        if s in ['low', '1']:
            return 1
        if s in ['high', '2']:
            return 2
        if s in ['very high', 'very_high', 'veryhigh', '3']:
            return 3
        raise ValueError(f"Unknown label value: {v}")

    for jf in sorted(json_files):
        path = os.path.join(annotations_dir, jf)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for k, v in data.items():
            label_map[str(k)] = to_int(v)

    return label_map

def subject_split(video_ids: List[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 13):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    subj = {}
    for vid in video_ids:
        s = infer_subject_id(vid)
        subj.setdefault(s, []).append(vid)

    subjects = sorted(subj.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    train_s = set(subjects[:n_train])
    val_s = set(subjects[n_train:n_train+n_val])
    test_s = set(subjects[n_train+n_val:])

    train_ids, val_ids, test_ids = [], [], []
    for s, vids in subj.items():
        if s in train_s: train_ids.extend(vids)
        elif s in val_s: val_ids.extend(vids)
        else: test_ids.extend(vids)

    return train_ids, val_ids, test_ids, {'train_subjects': len(train_s), 'val_subjects': len(val_s), 'test_subjects': len(test_s)}
