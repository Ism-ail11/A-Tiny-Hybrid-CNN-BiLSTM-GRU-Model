#!/usr/bin/env bash
set -e

CONFIG=${1:-configs/default.yaml}
OUTDIR=${2:-runs/exp1}

python train.py --config ${CONFIG} --out ${OUTDIR}
python evaluate.py --config ${CONFIG} --ckpt ${OUTDIR}/best.pt --out ${OUTDIR}/eval_test --split test
python compress.py --config ${CONFIG} --ckpt ${OUTDIR}/best.pt --out ${OUTDIR}/compress --export_name tiny_hybrid
python bench_onnx.py --model ${OUTDIR}/compress/tiny_hybrid_fp32.onnx
python bench_onnx.py --model ${OUTDIR}/compress/tiny_hybrid_int8.onnx
