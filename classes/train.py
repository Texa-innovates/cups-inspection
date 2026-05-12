#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a one-class (good-only) anomaly detector using EfficientNet-B7 features.
Defaults:
  --train_dir captures/camera0
  --out       oneclass_b7.joblib
"""

import argparse, os, glob, joblib
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.neighbors import NearestNeighbors

# -------- Dataset --------
class ImageFolderNoLabel(Dataset):
    def __init__(self, root, img_size=600):
        self.paths = []
        exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp")
        for e in exts:
            self.paths += glob.glob(os.path.join(root, e))
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root}")

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])

    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        im = Image.open(p).convert("RGB")
        return self.tf(im), p

# -------- EfficientNet-B7 Feature Extractor --------
class EffB7_Feature(nn.Module):
    def __init__(self, device):
        super().__init__()
        m = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        self.backbone = m.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # [B, 2560]
        return x

@torch.no_grad()
def compute_embeddings(loader, model, device):
    embs, paths = [], []
    for imgs, batch_paths in tqdm(loader, desc="Extracting features"):
        imgs = imgs.to(device, non_blocking=True)
        e = model(imgs).cpu().numpy()
        embs.append(e)
        paths += list(batch_paths)
    embs = np.concatenate(embs, axis=0)
    return embs, paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default=r"/home/texa_innovates/chromo_gpu/captures_1", help="Folder with ONLY good images")
    ap.add_argument("--out", default="oneclass_heritage_new.joblib", help="Artifact path to save")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=600)
    ap.add_argument("--knn_k", type=int, default=5, help="k for kNN anomaly scoring")
    ap.add_argument("--percentile", type=float, default=95.0, help="threshold at this percentile of good scores")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ds = ImageFolderNoLabel(args.train_dir, img_size=args.img_size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=(device.type=="cuda"))

    model = EffB7_Feature(device).to(device)

    # 1) Embeddings
    embs, _ = compute_embeddings(dl, model, device)

    # 2) Normalize (z-score)
    mean = embs.mean(axis=0, keepdims=True)
    std  = embs.std(axis=0, keepdims=True) + 1e-8
    embs_z = (embs - mean) / std

    # 3) kNN
    k = min(args.knn_k, max(1, len(embs_z)-1))
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(embs_z)

    # 4) Scores & threshold
    dists, _ = knn.kneighbors(embs_z, n_neighbors=k, return_distance=True)
    scores = dists.mean(axis=1)
    thresh = float(np.percentile(scores, args.percentile))

    # 5) Save artifact
    artifact = {
        "model_name": "efficientnet_b7_imagenet",
        "img_size": args.img_size,
        "knn_k": k,
        "threshold": thresh,
        "embed_mean": mean.astype(np.float32),
        "embed_std": std.astype(np.float32),
        "knn": knn,
    }
    joblib.dump(artifact, args.out)
    print(f"\n✅ Saved: {args.out}")
    print(f"Train images: {len(embs_z)}")
    print(f"k = {k}, threshold@{args.percentile}% = {thresh:.6f}")

if __name__ == "__main__":
    main()
