#!/usr/bin/env python3
"""Automated focal-subject quality gates (L* tolerance, monolithic mask fills)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mural_tiler import (  # noqa: E402
    LABELS_ARCHIVE,
    PREVIEW_JOB_NAME,
    apply_image_prep,
    choose_layout,
    fit_to_wall,
    render_focal_sheet_crop,
    rgb_to_lab,
)
from scripts.rembrandt_focal_verify import focal_bbox  # noqa: E402

L_TOLERANCE = 5.0
MONOLITH_FRAC = 0.32


def _zone_l(rgb, mask, thresh=0.35):
    m = mask >= thresh if mask is not None else np.zeros(rgb.shape[:2], dtype=bool)
    if not np.any(m):
        return None
    lab = rgb_to_lab(rgb[m])
    return {"mean_L": round(float(lab[:, 0].mean()), 2), "min_L": round(float(lab[:, 0].min()), 2)}


def _preview_l(prev, mask, thresh=0.35):
    m = mask >= thresh if mask is not None else np.zeros(prev.shape[:2], dtype=bool)
    if not np.any(m):
        return None
    lab = rgb_to_lab(prev[m])
    return {"mean_L": round(float(lab[:, 0].mean()), 2)}


def load_context(job_dir: Path):
    meta = json.loads((job_dir / PREVIEW_JOB_NAME).read_text())
    prep = meta.get("prep") or "auto"
    img_path = Path(meta.get("argv", {}).get("image") or "")
    from PIL import Image, ImageOps

    src = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    src = apply_image_prep(src, prep)
    wall_w_in, wall_h_in = meta["wall_width_ft"] * 12, meta["wall_height_ft"] * 12
    ppi = float(meta["ppi"])
    scale = float(meta.get("argv", {}).get("scale") or 1.0)
    img, _, crop = fit_to_wall(src, wall_w_in, wall_h_in, ppi)
    rgb = np.asarray(img)
    data = np.load(job_dir / LABELS_ARCHIVE)
    labels = data["labels"]
    palette = data["palette"]
    face_mask = data.get("face_mask")
    eye_mask = data.get("eye_mask")
    mouth_mask = data.get("mouth_mask")
    prev = np.asarray(__import__("PIL").Image.open(job_dir / "preview.png"))
    return meta, rgb, prev, labels, palette, face_mask, eye_mask, mouth_mask, scale, wall_w_in, wall_h_in


def check_monolithic_focal(
    labels,
    mask,
    palette,
    orig_l=None,
    prev_l=None,
    frac_limit=MONOLITH_FRAC,
):
    """Fail when one label dominates a focal mask *and* preview is visibly wrong (dark fill or bad L*)."""
    if mask is None:
        return True, None
    m = mask >= 0.5
    if int(m.sum()) < 20:
        m = mask >= 0.35
    if not np.any(m):
        return True, None
    pix = labels[m]
    if len(pix) < 20:
        return True, None
    counts = np.bincount(pix)
    dom = int(counts.argmax())
    top_frac = float(counts.max() / len(pix))
    pal_l = rgb_to_lab(palette)[:, 0]
    dom_l = float(pal_l[dom])
    if top_frac <= frac_limit:
        return True, top_frac
    if orig_l is not None and prev_l is not None and abs(prev_l - orig_l) <= L_TOLERANCE:
        if dom_l >= pal_l.min() + 4 or top_frac < 0.9:
            return True, top_frac
    if orig_l is not None and dom_l <= pal_l.min() + 2.5 and float(orig_l) - dom_l > 10 and top_frac >= 0.4:
        return False, top_frac
    if orig_l is not None and prev_l is not None and abs(prev_l - orig_l) > L_TOLERANCE and top_frac >= 0.55:
        return False, top_frac
    if top_frac >= 0.88 and orig_l is not None and prev_l is not None and abs(prev_l - orig_l) > L_TOLERANCE:
        return False, top_frac
    return True, top_frac


MOSAIC_IOU_MIN = 0.5


def focal_bbox_from_meta(meta, shape):
    for key in ("subject_priority", "face_priority"):
        fp = meta.get(key) or {}
        if fp.get("eye_disks_work") or fp.get("mouth_ellipses_work"):
            box = focal_bbox(fp)
            if box is not None:
                return box
    return None


def _label_boundary_mask(labels, x0, y0, x1, y1):
    sub = labels[y0:y1, x0:x1]
    edge = np.zeros(sub.shape, dtype=bool)
    edge[1:, :] |= sub[1:, :] != sub[:-1, :]
    edge[:, 1:] |= sub[:, 1:] != sub[:, :-1]
    return edge


def _mosaic_outline_mask(crop, thresh=14):
    gray = np.dot(crop[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    ink = gray < 88
    grad = (gx + gy) > thresh
    return ink | grad


def mosaic_label_boundary_iou(crop, labels, box, dilate=3, match_px=12):
    """Overlap of dilated label-map edges with mosaic outline ink (tolerant of rasterization)."""
    x0, y0, x1, y1 = (int(v) for v in box)
    if crop is None or crop.shape[0] != y1 - y0 or crop.shape[1] != x1 - x0:
        return 0.0
    struct = ndimage.generate_binary_structure(2, 1)
    label_e = _label_boundary_mask(labels, x0, y0, x1, y1)
    label_e = ndimage.binary_dilation(label_e, iterations=dilate, structure=struct)
    mosaic_e = _mosaic_outline_mask(crop)
    mosaic_e = ndimage.binary_dilation(mosaic_e, iterations=dilate, structure=struct)
    band = mosaic_e
    if match_px > 0:
        dt = ndimage.distance_transform_edt(~mosaic_e)
        band = band | (dt <= match_px)
    inter = float(np.logical_and(label_e, band).sum())
    union = float(np.logical_or(label_e, band).sum())
    if union <= 0 or label_e.sum() <= 0:
        return 0.0
    jaccard = inter / union
    recall = inter / float(label_e.sum())
    return max(jaccard, recall)


def _digitish_blobs(gray, region):
    blob = ((gray < 110) & region).astype(np.int32)
    labels_cc, nlab = ndimage.label(blob)
    count = 0
    for lid in range(1, int(nlab) + 1):
        area = int((labels_cc == lid).sum())
        if 12 <= area <= 2500:
            count += 1
    return count


def check_focal_sheet_crop(crop, eye_mask, box, mouth_mask=None, labels=None):
    """Sheet column must show real PDF linework and number-like strokes in the eye area."""
    if crop is None or crop.size == 0:
        return ["missing focal sheet crop"]
    fails = []
    gray = np.dot(crop[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edge_px = int(((gx + gy) > 14).sum())
    if edge_px < 120:
        fails.append(f"sheet crop lacks outlines ({edge_px} edge px)")
    if box is not None:
        x0, y0, x1, y1 = (int(v) for v in box)
        region = np.ones(gray.shape, dtype=bool)
        if eye_mask is not None or mouth_mask is not None:
            region = np.zeros(gray.shape, dtype=bool)
            if eye_mask is not None:
                region |= eye_mask[y0:y1, x0:x1] >= 0.35
            if mouth_mask is not None:
                region |= mouth_mask[y0:y1, x0:x1] >= 0.35
            if np.any(region):
                region = ndimage.binary_dilation(region, iterations=16)
        digits = _digitish_blobs(gray, region)
        if digits < 1:
            fails.append("no number-like strokes in eye sheet crop")
    if labels is not None and box is not None:
        iou = mosaic_label_boundary_iou(crop, labels, box)
        if iou < MOSAIC_IOU_MIN:
            fails.append(f"sheet mosaic outline IoU {iou:.2f} < {MOSAIC_IOU_MIN}")
    return fails


def check_l_tolerance(orig_stats, prev_stats, tol=L_TOLERANCE):
    if not orig_stats or not prev_stats:
        return True, None
    err = float(prev_stats["mean_L"] - orig_stats["mean_L"])
    return abs(err) <= tol, err


def audit_job(job_dir: Path, check_sheet_crop=False):
    meta, rgb, prev, labels, palette, face_mask, eye_mask, mouth_mask, scale, w_in, h_in = load_context(job_dir)
    eye_o = _zone_l(rgb, eye_mask)
    mouth_o = _zone_l(rgb, mouth_mask)
    skin_m = None
    if face_mask is not None:
        eye = eye_mask >= 0.35 if eye_mask is not None else False
        mouth = mouth_mask >= 0.35 if mouth_mask is not None else False
        skin_m = (face_mask >= 0.35) & ~(eye if isinstance(eye, np.ndarray) else False) & ~(
            mouth if isinstance(mouth, np.ndarray) else False
        )
    skin_o = _zone_l(rgb, skin_m.astype(np.float32) if skin_m is not None else None)
    eye_p = _preview_l(prev, eye_mask)
    mouth_p = _preview_l(prev, mouth_mask)
    skin_p = _preview_l(prev, skin_m.astype(np.float32) if skin_m is not None else None)
    failures = []
    for zone, o, p in (("eye", eye_o, eye_p), ("mouth", mouth_o, mouth_p), ("skin", skin_o, skin_p)):
        ok, err = check_l_tolerance(o, p)
        if not ok:
            failures.append(f"{zone} L* error {err:+.1f} (tol {L_TOLERANCE})")
    if mouth_mask is not None:
        ok, frac = check_monolithic_focal(
            labels,
            mouth_mask,
            palette,
            (mouth_o or {}).get("mean_L"),
            (mouth_p or {}).get("mean_L"),
        )
        if not ok:
            failures.append(f"mouth monolithic label {frac:.0%} of mask")
    if eye_mask is not None:
        ok, frac = check_monolithic_focal(
            labels,
            eye_mask,
            palette,
            (eye_o or {}).get("mean_L"),
            (eye_p or {}).get("mean_L"),
        )
        if not ok:
            failures.append(f"eye monolithic label {frac:.0%} of mask")
    if check_sheet_crop and eye_mask is not None and np.any(eye_mask >= 0.2):
        overlap = float(meta.get("argv", {}).get("overlap_in") or 0.5)
        layout = choose_layout(w_in * scale, h_in * scale, overlap)
        job = {
            "scale": scale,
            "ppi": float(meta["ppi"]),
            "sheet_tint": float(meta.get("argv", {}).get("sheet_tint", 0)),
        }
        box = focal_bbox_from_meta(meta, labels.shape)
        if box is None:
            failures.append("missing focal bbox for sheet mosaic")
        else:
            mural_w, mural_h = w_in * scale, h_in * scale
            crop = render_focal_sheet_crop(
                job_dir, layout, job, box, eye_mask, mouth_mask, face_mask, mural_w=mural_w, mural_h=mural_h,
            )
            failures.extend(check_focal_sheet_crop(crop, eye_mask, box, mouth_mask, labels))
    return failures


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("usage: focal_quality.py JOB_DIR ...", file=sys.stderr)
        return 2
    failed = False
    for job_dir in paths:
        fails = audit_job(job_dir)
        tag = job_dir.name
        if fails:
            failed = True
            print(f"FAIL {tag}: " + "; ".join(fails))
        else:
            print(f"OK {tag}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())