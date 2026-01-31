from __future__ import annotations
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .splits import load_attention_labels

class DAiSEEClipDataset(Dataset):
    """Loads DAiSEE clips as fixed-length frame tensors (T, 3, H, W).
    - frame sampling at target_fps (approx; uses uniform sampling over the video)
    - adaptive luminance normalization per frame
    """

    def __init__(
        self,
        root: str,
        videos_dir: str,
        annotations_dir: str,
        ids: list[str],
        num_frames: int = 30,
        img_size: int = 224,
        brightness_target: float = 128.0,
        brightness_strength: float = 0.7,
        normalize_imagenet: bool = True,
    ):
        self.root = root
        self.videos_path = os.path.join(root, videos_dir)
        self.annotations_path = os.path.join(root, annotations_dir)
        self.ids = ids
        self.num_frames = num_frames
        self.img_size = img_size
        self.brightness_target = brightness_target
        self.brightness_strength = brightness_strength

        self.labels = load_attention_labels(self.annotations_path)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        t = [transforms.ToTensor(),
             transforms.Resize((img_size, img_size), antialias=True)]
        if normalize_imagenet:
            t.append(transforms.Normalize(mean=mean, std=std))
        self.tf = transforms.Compose(t)

    def __len__(self):
        return len(self.ids)

    def _adaptive_brightness(self, bgr: np.ndarray) -> np.ndarray:
        # Convert BGR->YCrCb and shift luminance toward target (gentle)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        y = ycrcb[..., 0]
        cur = float(np.mean(y))
        if cur < 1e-6:
            return bgr
        delta = (self.brightness_target - cur) * self.brightness_strength
        y = np.clip(y + delta, 0, 255)
        ycrcb[..., 0] = y
        out = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
        return out

    def _read_video(self, path: str) -> list[np.ndarray]:
        cap = cv2.VideoCapture(path)
        frames = []
        if not cap.isOpened():
            cap.release()
            return frames
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)  # BGR
        cap.release()
        return frames

    def _sample_indices(self, n: int) -> np.ndarray:
        # Uniform sampling of num_frames from n frames; pad by repetition
        if n <= 0:
            return np.zeros((self.num_frames,), dtype=np.int64)
        if n >= self.num_frames:
            return np.linspace(0, n-1, self.num_frames).round().astype(np.int64)
        # pad: repeat last index
        idx = np.linspace(0, n-1, n).round().astype(np.int64)
        pad = np.full((self.num_frames - n,), n-1, dtype=np.int64)
        return np.concatenate([idx, pad], axis=0)

    def __getitem__(self, i: int):
        vid = self.ids[i]
        # locate video file: try common extensions
        candidates = [os.path.join(self.videos_path, vid + ext) for ext in [".mp4", ".avi", ".mkv"]]
        candidates += [os.path.join(self.videos_path, vid)]
        video_path = None
        for c in candidates:
            if os.path.exists(c):
                video_path = c
                break
        if video_path is None:
            raise FileNotFoundError(f"Video not found for id={vid} under {self.videos_path}")

        frames = self._read_video(video_path)
        idxs = self._sample_indices(len(frames))

        clip = []
        for j in idxs:
            frame = frames[int(j)] if len(frames) else np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            frame = self._adaptive_brightness(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # torchvision transforms expect PIL or torch; we go via torch tensor later
            # Use ToTensor after converting to uint8 RGB
            clip.append(frame)

        clip = np.stack(clip, axis=0)  # (T, H, W, 3)
        clip_t = torch.stack([self.tf(clip[t]) for t in range(clip.shape[0])], dim=0)  # (T,3,H,W)

        if vid not in self.labels:
            raise KeyError(f"No label found for video id={vid} in {self.annotations_path}")
        y = int(self.labels[vid])
        return clip_t, y, vid
