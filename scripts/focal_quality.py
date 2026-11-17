#!/usr/bin/env python3
"""Automated focal-subject quality gates (L* tolerance, monolithic mask fills)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mural_tiler import (  # noqa: E402
    LABELS_ARCHIVE,
    PREVIEW_JOB_NAME,
    apply_image_prep,
    fit_to_wall,
    rgb_to_lab,
)

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


def check_l_tolerance(orig_stats, prev_stats, tol=L_TOLERANCE):
    if not orig_stats or not prev_stats:
        return True, None
    err = float(prev_stats["mean_L"] - orig_stats["mean_L"])
    return abs(err) <= tol, err


def audit_job(job_dir: Path):
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