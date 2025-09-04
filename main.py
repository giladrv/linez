# Standard
import argparse
import os
# External
import cv2 as cv
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, SamPredictor
import torch
# Internal
from utils import (
    SAM_MODELS,
    parse_points,
    save_masks,
    draw_overlay,
    load_model,
    build_wall_polygon_and_filter,
    masks_exist,
    load_saved_masks,
)

def draw_wall_and_holds(img_bgr, wall_poly, kept_masks, alpha = 0.35, wall_color = (0,255,255), hold_color = (255,0,0)):
    """
    Draw wall polygon fill + outline in yellow, and hold outlines in blue.
    """
    import cv2 as cv
    import numpy as np
    vis = img_bgr.copy()

    # wall fill + outline
    if wall_poly is not None and len(wall_poly) >= 3:
        wall_mask = np.zeros(img_bgr.shape[:2], np.uint8)
        cv.fillPoly(wall_mask, [np.asarray(wall_poly, np.int32)], 255)
        wall_rgb = np.zeros_like(img_bgr)
        wall_rgb[:] = wall_color
        # Blend globally, then write back only where wall_mask > 0
        blended = cv.addWeighted(vis, 1.0, wall_rgb, alpha, 0)
        vis[wall_mask > 0] = blended[wall_mask > 0]
        cv.polylines(vis, [np.asarray(wall_poly, np.int32)], True, wall_color, 3)

    # holds outlines in blue
    for m in kept_masks:
        seg = (m["segmentation"].astype(np.uint8) * 255)
        cnts, _ = cv.findContours(seg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv.drawContours(vis, cnts, -1, hold_color, 2)
    return vis


def draw_wall_components_overlay(img_bgr, masks, wall_component_indices, wall_poly, alpha = 0.35, wall_color = (0,255,255)):
    """
    Debug overlay showing segments used as wall components.
    Fills the union of component masks in yellow and outlines the final wall polygon.
    """
    import cv2 as cv
    import numpy as np
    vis = img_bgr.copy()
    if wall_component_indices:
        H, W = img_bgr.shape[:2]
        union = np.zeros((H, W), np.uint8)
        for i in wall_component_indices:
            seg = (masks[i]["segmentation"].astype(np.uint8) * 255)
            union = cv.bitwise_or(union, seg)
        # Blend yellow over union mask only
        color = np.zeros_like(img_bgr)
        color[:] = wall_color
        blended = cv.addWeighted(vis, 1.0, color, alpha, 0)
        vis[union > 0] = blended[union > 0]
        # Draw contours of components
        cnts, _ = cv.findContours(union, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv.drawContours(vis, cnts, -1, wall_color, 2)
    if wall_poly is not None and len(wall_poly) >= 3:
        cv.polylines(vis, [np.asarray(wall_poly, np.int32)], True, wall_color, 3)
    return vis

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required = True, help = "input image")
    ap.add_argument("--model", default = "vit_h", choices = SAM_MODELS.keys())
    ap.add_argument("--mode", default = "auto", choices = ["auto","points","box"])
    ap.add_argument("--points", nargs = "*", default = [], help = 'e.g. --points "640,480:1" "700,520:0"')
    ap.add_argument("--box", default = "", help = "x1,y1,x2,y2")
    ap.add_argument("--out", default = None)
    ap.add_argument("--device", default = "cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Resolve image path: prefer given path; fall back to .temp/<image>
    img_path = args.image
    if not os.path.isfile(img_path):
        alt_path = os.path.join('.temp', args.image)
        if os.path.isfile(alt_path):
            img_path = alt_path
    img_bgr = cv.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {args.image}")

    sam = load_model(args.model, args.device)

    # Prepare output dir early and prefer loading existing masks
    out_dir = os.path.join('.temp', args.out or os.path.splitext(os.path.basename(args.image))[0])
    if masks_exist(out_dir):
        print(f"Found existing masks in {out_dir}; loading from disk...")
        try:
            masks = load_saved_masks(out_dir)
        except Exception as e:
            print(f"[WARN] Failed to load saved masks: {e}. Regenerating...")
            masks = None
    else:
        masks = None

    if masks is None:
        if args.mode == "auto":
            print("Running automatic mask generation...")
            gen = SamAutomaticMaskGenerator(
                sam,
                points_per_side = 32,          # denser = more masks, slower
                pred_iou_thresh = 0.88,
                stability_score_thresh = 0.92,
                box_nms_thresh = 0.6,
                min_mask_region_area = 64      # remove tiny specks
            )
            masks = gen.generate(img_bgr)
        else:
            if args.mode == "points" and len(args.points) == 0:
                raise ValueError("No points provided for --mode points")
            if args.mode == "box" and args.box == "":
                raise ValueError("No box provided for --mode box")
            print(f"Running {args.mode}-based prediction...")
            predictor = SamPredictor(sam)
            predictor.set_image(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
            masks = []
            if args.mode == "points":
                pts, lbls = parse_points(args.points)
                m, _, _ = predictor.predict(point_coords = pts, point_labels = lbls, multimask_output = True)
                for seg in m:
                    masks.append({"segmentation": seg})
            elif args.mode == "box":
                x1,y1,x2,y2 = map(int, args.box.split(","))
                m, _, _ = predictor.predict(box = np.array([x1,y1,x2,y2]), multimask_output = True)
                for seg in m:
                    masks.append({"segmentation": seg})
        # Save masks only when just generated
        os.makedirs(out_dir, exist_ok = True)
        save_masks(out_dir, masks)

    # ---- Wall polygon pipeline (strict) ----
    try:
        kept_masks, wall_poly, wall_mask, wall_components, kept_indices = build_wall_polygon_and_filter(
            masks, img_bgr.shape[:2], hold_inside_frac = 0.8
        )
    except ValueError as e:
        print(f"[WARN] {e}")
        # Save a diagnostic empty mask and continue with the raw overlay
        wall_poly, wall_mask, kept_masks, kept_indices, wall_components = None, np.zeros(img_bgr.shape[:2], np.uint8), [], [], []

    # Ensure output dir exists for overlays
    os.makedirs(out_dir, exist_ok = True)

    # Save standard overlay and masks bundle
    save_masks(out_dir, masks)
    overlay = draw_overlay(img_bgr, masks)
    overlay_file = os.path.join(out_dir, 'overlay.png')
    cv.imwrite(overlay_file, overlay)

    # Save wall mask and a color-coded overlay (yellow wall, blue holds)
    wall_mask_file = os.path.join(out_dir, 'wall_mask.png')
    cv.imwrite(wall_mask_file, wall_mask)

    # Exclude wall component masks from holds
    hold_indices = [i for i in kept_indices if i not in set(wall_components)]
    # Also remove any remaining big segments (>= 1% of image) from holds
    H, W = img_bgr.shape[:2]
    img_area = float(H * W)
    min_hold_area_frac = 0.01
    small_hold_indices = []
    for i in hold_indices:
        seg = masks[i]["segmentation"]
        area = int(np.count_nonzero(seg))
        if area < (min_hold_area_frac * img_area):
            small_hold_indices.append(i)
    hold_masks = [masks[i] for i in small_hold_indices]
    color_overlay = draw_wall_and_holds(img_bgr, wall_poly, hold_masks)
    wall_overlay_file = os.path.join(out_dir, 'wall_holds_overlay.png')
    cv.imwrite(wall_overlay_file, color_overlay)

    # Save debug overlay highlighting wall components used to build the wall
    comp_overlay = draw_wall_components_overlay(img_bgr, masks, wall_components, wall_poly)
    wall_comps_overlay_file = os.path.join(out_dir, 'wall_components_overlay.png')
    cv.imwrite(wall_comps_overlay_file, comp_overlay)

    print(f"Saved overlay: {overlay_file}")
    print(f"Saved wall mask: {wall_mask_file}")
    print(f"Saved wall+holds overlay: {wall_overlay_file}")
    print(f"Saved wall components overlay: {wall_comps_overlay_file}")
    print(f"Saved {len(masks)} mask PNGs + JSON: {out_dir}")

if __name__ == "__main__":
    main()
