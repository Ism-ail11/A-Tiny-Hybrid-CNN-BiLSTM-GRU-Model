import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_reliability(probs, y_true, out_path, n_bins=15):
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins+1)

    accs, confs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            accs.append(0.0); confs.append((lo+hi)/2); counts.append(0)
        else:
            accs.append(float((pred[mask] == y_true[mask]).mean()))
            confs.append(float(conf[mask].mean()))
            counts.append(int(mask.sum()))

    plt.figure()
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(confs, accs, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability diagram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
