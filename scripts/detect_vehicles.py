"""Real detector boxes for every frame: torchvision Faster R-CNN ResNet50-FPN v2 (COCO), vehicle classes only.

Replaces DeepSense's annotation boxes (an oracle detector) and gives boxes for scenarios without
annotations (31-35). Full-resolution images from the extracted scenario folders; COCO classes car (3),
motorcycle (4), bus (6), truck (8); score >= --min_score.

Writes data/cache/<scenario>_det_boxes.npz:
  xs (N, M) box x-centre / width in [0, 1], NaN padded, sorted by score
  ys, ws, hs (N, M) same layout, scores (N, M), cls (N, M) COCO label ids (-1 padded)

    python scripts/detect_vehicles.py --scenarios scenario1 scenario2 ... --bs 8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tta_beam.data import CACHE, _col, find_csv  # noqa: E402

VEHICLES = (3, 4, 6, 8)


def main():
    import pandas as pd
    from PIL import Image
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
    from torchvision.transforms.functional import pil_to_tensor
    from tqdm import tqdm

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--min_score", type=float, default=0.5)
    ap.add_argument("--max_boxes", type=int, default=12)
    a = ap.parse_args()

    dev = "cuda"
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=a.min_score).to(dev).eval()
    prep = weights.transforms()

    for name in a.scenarios:
        out = CACHE / f"{name}_det_boxes.npz"
        if out.exists():
            print("cached", out.name, flush=True)
            continue
        csv = find_csv(name)
        df = pd.read_csv(csv)
        paths = [csv.parent / str(p).lstrip("./") for p in df[_col(df, "rgb")]]
        N, M = len(paths), a.max_boxes
        xs, ys, ws, hs, sc = (np.full((N, M), np.nan, np.float32) for _ in range(5))
        cl = np.full((N, M), -1, np.int16)
        for i in tqdm(range(0, N, a.bs), desc=name, mininterval=30):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + a.bs]]
            W, H = imgs[0].size
            batch = [prep(pil_to_tensor(im)).to(dev) for im in imgs]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                dets = model(batch)
            for j, det in enumerate(dets):
                keep = torch.isin(det["labels"], torch.tensor(VEHICLES, device=dev)) & (det["scores"] >= a.min_score)
                b, s, l = det["boxes"][keep].float().cpu().numpy(), det["scores"][keep].float().cpu().numpy(), det["labels"][keep].cpu().numpy()
                order = np.argsort(-s)[:M]
                b, s, l = b[order], s[order], l[order]
                k = len(s)
                xs[i + j, :k] = (b[:, 0] + b[:, 2]) / 2 / W
                ys[i + j, :k] = (b[:, 1] + b[:, 3]) / 2 / H
                ws[i + j, :k] = (b[:, 2] - b[:, 0]) / W
                hs[i + j, :k] = (b[:, 3] - b[:, 1]) / H
                sc[i + j, :k] = s
                cl[i + j, :k] = l
        np.savez_compressed(out, xs=xs, ys=ys, ws=ws, hs=hs, scores=sc, cls=cl)
        print(f"wrote {out.name}: {N} frames, >=1 vehicle in {np.isfinite(xs[:, 0]).mean():.2f}, "
              f"mean {np.isfinite(xs).sum(1).mean():.2f} per frame", flush=True)


if __name__ == "__main__":
    main()
