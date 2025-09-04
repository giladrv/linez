# Standard
import json
import os
from pathlib import Path
# Extenral
import cv2 as cv
import numpy as np
import requests
from segment_anything import sam_model_registry
from tqdm import tqdm
# Internal

SAM_WEIGHTS_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything/"
SAM_MODELS = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}

WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok = True)

def colorize(mask):
    c = np.random.default_rng().integers(0, 255, size = 3, dtype = np.uint8)
    out = np.zeros((*mask.shape, 3), np.uint8)
    out[mask > 0] = c
    return out

def draw_overlay(img_bgr, masks, alpha = 0.45, outline_px = 2):
    overlay = img_bgr.copy()
    for m in masks:
        seg = m["segmentation"].astype(np.uint8) * 255
        color = colorize(seg)
        overlay = cv.addWeighted(overlay, 1.0, color, alpha, 0)
        cnts, _ = cv.findContours(seg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(overlay, cnts, -1, (255, 0, 0), outline_px)
    return overlay

def save_masks(out_dir, masks):
    mask_dir = os.path.join(out_dir, 'masks')
    os.makedirs(mask_dir, exist_ok = True)
    meta = []
    for i, m in enumerate(masks):
        seg = (m["segmentation"].astype(np.uint8) * 255)
        mask_file = os.path.join(mask_dir, f"{i:04d}.png")
        cv.imwrite(mask_file, seg)
        meta.append({
            "id": i,
            "area": int(m["area"]),
            "bbox": [int(x) for x in m["bbox"]],
            "predicted_iou": float(m.get("predicted_iou", 0.0)),
            "stability_score": float(m.get("stability_score", 0.0)),
            "mask_path": str(mask_file),
        })
    with open(os.path.join(out_dir, "masks.json"), "w") as f:
        json.dump({"masks": meta}, f, indent = 2)
    return out_dir

def parse_points(arg_list):
    # format: "x,y:label" label 1 = foreground, 0 = background
    pts, lbls = [], []
    for s in arg_list:
        xy, lab = s.split(":")
        x, y = map(int, xy.split(","))
        pts.append([x, y])
        lbls.append(int(lab))
    return np.array(pts), np.array(lbls)

def download_sam(model = "vit_h", out_file = None):
    if model not in SAM_MODELS:
        raise ValueError(f"Unknown model {model}, choose from {list(SAM_MODELS)}")
    url = SAM_WEIGHTS_BASE_URL + SAM_MODELS[model]
    if out_file is None:
        out_file = SAM_MODELS[model]

    dest_path = f"{WEIGHTS_DIR}/{out_file}"
    print(f"Downloading {model} weights to {dest_path}...")

    with requests.get(url, stream = True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total = total, unit = "B", unit_scale = True, unit_divisor = 1024
        ) as bar:
            for chunk in r.iter_content(chunk_size = 8192):
                if chunk:  # skip keep-alive
                    f.write(chunk)
                    bar.update(len(chunk))

def load_model(model_name, device):
    checkpoint = f"{WEIGHTS_DIR}/{SAM_MODELS[model_name]}"
    if not os.path.isfile(checkpoint):
        download_sam(model_name, out_file = SAM_MODELS[model_name])
    print(f"Loading model {model_name} from {checkpoint} on {device}")
    sam = sam_model_registry[model_name](checkpoint = checkpoint)
    sam.to(device)
    return sam
