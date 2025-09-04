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

def masks_exist(out_dir):
    """Return True if a previous masks export exists in out_dir."""
    json_path = os.path.join(out_dir, "masks.json")
    mask_dir = os.path.join(out_dir, "masks")
    return os.path.isfile(json_path) and os.path.isdir(mask_dir)

def _resolve_mask_path(out_dir, mask_path):
    """Resolve saved mask path robustly across relative/absolute variants."""
    candidates = [
        mask_path,
        os.path.join(out_dir, mask_path),
        os.path.join(out_dir, "masks", os.path.basename(mask_path)),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return mask_path  # fall back; caller may fail on read

def load_saved_masks(out_dir):
    """Load masks list from out_dir/masks.json and mask PNGs in out_dir/masks.

    Returns a list of dicts compatible with SAM's output where
    entry["segmentation"] is a boolean array.
    """
    json_path = os.path.join(out_dir, "masks.json")
    with open(json_path, "r") as f:
        meta = json.load(f)["masks"]
    loaded = []
    for m in meta:
        mp = _resolve_mask_path(out_dir, m.get("mask_path", ""))
        seg_img = cv.imread(mp, cv.IMREAD_GRAYSCALE)
        if seg_img is None:
            raise FileNotFoundError(f"Mask image not found: {mp}")
        seg = seg_img > 0
        loaded.append({
            "segmentation": seg,
            "area": m.get("area"),
            "bbox": m.get("bbox"),
            "predicted_iou": m.get("predicted_iou"),
            "stability_score": m.get("stability_score"),
        })
    return loaded

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


# ===================== Wall post-processing helpers (largest SAM mask, strict 50%) =====================

def polygon_from_mask(mask: np.ndarray, prefer_vertices: int = 4):
    """Approximate the outer contour of a binary mask as a polygon with ~prefer_vertices vertices."""
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key = cv.contourArea)
    hull = cv.convexHull(c)
    peri = cv.arcLength(hull, True)
    eps = 0.01
    approx = cv.approxPolyDP(hull, eps * peri, True)
    tries = 0
    while tries < 25 and len(approx) != prefer_vertices and len(approx) >= 3:
        if len(approx) > prefer_vertices:
            eps += 0.005
        else:
            eps = max(0.005, eps - 0.002)
        approx = cv.approxPolyDP(hull, eps * peri, True)
        tries += 1
    return approx.reshape(-1, 2)

def rasterize_polygon(poly, shape):
    """Create a filled mask from polygon points for an output image size (H, W)."""
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    if poly is None or len(poly) < 3:
        return m
    cv.fillPoly(m, [np.asarray(poly, np.int32)], 255)
    return m

def largest_sam_mask(masks, image_shape):
    """Return (index, mask, area_fraction). 'mask' is uint8 0/255."""
    H, W = image_shape
    img_area = float(H * W)
    if not masks:
        return None, None, 0.0
    areas = []
    segs = []
    for m in masks:
        seg = (m["segmentation"].astype(np.uint8)) * 255
        segs.append(seg)
        areas.append(int(cv.countNonZero(seg)))
    idx = int(np.argmax(areas))
    area = areas[idx]
    frac = area / img_area
    return idx, segs[idx], frac

def extend_wall_with_segments(main_poly, seg_polys, img_shape, edge_tol_px: int = 12, min_area_frac: float = 0.01, min_overlap_frac: float = 0.20):
    """
    Extend main wall polygon with large adjacent polygons whose edges substantially overlap the main wall edges.
    Returns (extended_polygon, extended_mask, accepted_segment_polys).
    """
    H, W = img_shape
    main_mask = rasterize_polygon(main_poly, (H, W))
    boundary = cv.Canny(main_mask, 50, 150)
    # distance to the wall boundary; small = close to an edge of the wall
    dist = cv.distanceTransform(255 - boundary, cv.DIST_L2, 3)
    accept_masks = [main_mask.copy()]
    accepted_polys = []
    accepted_idx = []
    img_area = float(H * W)

    for i, poly in enumerate(seg_polys):
        if poly is None or len(poly) < 3:
            continue
        pmask = rasterize_polygon(poly, (H, W))
        # area criterion: at least 1% of the image
        if pmask.sum() / 255.0 < (min_area_frac * img_area):
            continue
        # boundary-overlap criterion
        cnts, _ = cv.findContours(cv.Canny(pmask, 50, 150), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        pts = cnts[0].reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        dvals = dist[pts[:, 1], pts[:, 0]]
        frac_close = float((dvals <= edge_tol_px).sum()) / float(len(dvals))
        if frac_close >= min_overlap_frac:
            accept_masks.append(pmask)
            accepted_polys.append(poly)
            accepted_idx.append(i)

    union = np.clip(np.sum(np.stack(accept_masks, axis = 0), axis = 0), 0, 255).astype(np.uint8)
    union[union > 0] = 255
    cnts, _ = cv.findContours(union, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return main_poly, main_mask, accepted_polys
    uc = max(cnts, key = cv.contourArea)
    hull = cv.convexHull(uc)
    hull_poly = hull.reshape(-1, 2)
    hull_mask = np.zeros_like(union)
    cv.fillPoly(hull_mask, [hull], 255)
    return hull_poly, hull_mask, accepted_idx

def filter_masks_by_wall(masks, wall_mask, inside_frac: float = 0.5):
    """Return indices of SAM masks with at least 'inside_frac' of their area inside the wall mask."""
    kept_idx = []
    wall_bin = (wall_mask > 0).astype(np.uint8)
    for i, m in enumerate(masks):
        seg = (m["segmentation"].astype(np.uint8) * 255)
        inter = cv.bitwise_and(seg, seg, mask = wall_bin)
        inside = int(cv.countNonZero(inter))
        total = int(cv.countNonZero(seg))
        frac = (inside / total) if total > 0 else 0.0
        if frac >= inside_frac:
            kept_idx.append(i)
    return kept_idx

def build_wall_polygon_and_filter(masks, image_shape, min_main_frac: float = 0.5, edge_tol_px: int = 12, min_overlap_frac: float = 0.20, hold_inside_frac: float = 0.5):
    """
    Steps:
      1) Determine largest SAM mask. Must be >= min_main_frac of the image; otherwise raise ValueError.
      2) Approximate as polygon.
      3) Find other segments that are big polygons (>= 1% of image) and whose boundary overlaps the wall's boundary; extend the wall polygon with them.
      4) Toss out all masks that are < hold_inside_frac inside the final wall polygon.
    Returns: (kept_masks, wall_polygon, wall_mask, wall_component_polys)
    """
    H, W = image_shape[:2]
    # 1) largest mask
    idx, largest_mask, frac = largest_sam_mask(masks, (H, W))
    if largest_mask is None or frac < min_main_frac:
        raise ValueError(f"Largest SAM mask too small: {frac:.3f} < {min_main_frac:.3f} of image")

    # 2) polygon
    main_poly = polygon_from_mask(largest_mask, prefer_vertices = 4)

    # 3) extend with adjacent big polygons
    seg_polys = []
    for m in masks:
        seg = (m["segmentation"].astype(np.uint8) * 255)
        p = polygon_from_mask(seg, prefer_vertices = 4)
        seg_polys.append(p)

    wall_poly, wall_mask, accepted_idx = extend_wall_with_segments(
        main_poly, seg_polys, (H, W),
        edge_tol_px = edge_tol_px, min_area_frac = 0.01, min_overlap_frac = min_overlap_frac
    )

    # 4) filter masks using the requested inside fraction for holds
    kept_indices = filter_masks_by_wall(masks, wall_mask, inside_frac = hold_inside_frac)
    kept = [masks[i] for i in kept_indices]
    # Return indices of wall components: include the main mask idx and accepted adjacency idxs
    wall_component_indices = sorted({idx} | set(accepted_idx))
    return kept, wall_poly, wall_mask, wall_component_indices, kept_indices
