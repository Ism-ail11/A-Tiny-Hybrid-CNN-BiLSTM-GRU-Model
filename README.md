# Tiny Hybrid CNN–BiLSTM–GRU (DAiSEE) — Training → Compression → Evaluation

This repo provides an end-to-end, **vision-only** pipeline for **4-level attention recognition** from short webcam clips (DAiSEE).
It includes: preprocessing (frame sampling + adaptive luminance normalization), a compact **CNN–BiLSTM–GRU** model (3-branch),
training, evaluation (accuracy, macro-F1, QWK, ordinal MAE, calibration), and deployment-oriented compression (structured pruning,
INT8 quantization, temperature scaling) with ONNX export.

