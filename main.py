# Standard
import argparse
import os
# External
import cv2 as cv
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, SamPredictor
import torch
# Internal
from utils import SAM_MODELS, parse_points, save_masks, draw_overlay, load_model

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

    img_bgr = cv.imread(".temp/" + args.image)
    if img_bgr is None:
        raise FileNotFoundError(args.image)

    sam = load_model(args.model, args.device)

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

    out_dir = os.path.join('.temp', args.out or os.path.splitext(os.path.basename(args.image))[0])
    save_masks(out_dir, masks)
    overlay = draw_overlay(img_bgr, masks)
    out_file = os.path.join(out_dir, 'overlay.png')
    cv.imwrite(out_file, overlay)
    print(f"Saved overlay: {out_file}")
    print(f"Saved {len(masks)} mask PNGs + JSON: {out_dir}")

if __name__ == "__main__":
    main()
