from __future__ import annotations
import argparse, time, os
import numpy as np
import onnxruntime as ort

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--num_frames', type=int, default=30)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--runs', type=int, default=500)
    ap.add_argument('--warmup', type=int, default=50)
    return ap.parse_args()

def main():
    args = parse_args()
    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    inp_name = sess.get_inputs()[0].name

    x = np.random.randn(1, args.num_frames, 3, args.img_size, args.img_size).astype(np.float32)

    for _ in range(args.warmup):
        _ = sess.run(None, {inp_name: x})

    t0 = time.time()
    for _ in range(args.runs):
        _ = sess.run(None, {inp_name: x})
    t1 = time.time()
    avg_ms = (t1 - t0) * 1000.0 / args.runs
    thr = 1000.0 / avg_ms
    print(f"avg_latency_ms={avg_ms:.3f}  throughput_clips_per_s={thr:.2f}")

if __name__ == '__main__':
    main()
