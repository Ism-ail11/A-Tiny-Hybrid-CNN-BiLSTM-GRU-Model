from __future__ import annotations
import numpy as np

def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())

def ordinal_mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    return float(np.mean(np.abs(y_true - y_pred)))

def ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error for multi-class probs.
    probs: (N, C), y_true: (N,)
    """
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        acc_bin = np.mean(pred[mask] == y_true[mask])
        conf_bin = np.mean(conf[mask])
        ece_val += (np.sum(mask) / n) * abs(acc_bin - conf_bin)
    return float(ece_val)

def brier_multi(probs: np.ndarray, y_true: np.ndarray, num_classes: int) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    onehot = np.zeros((len(y_true), num_classes), dtype=np.float64)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
