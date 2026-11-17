#!/usr/bin/env python3
"""Hard regression gates for kitten_ragdoll_canvas.jpg 8x10 portrait medium builds."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mural_tiler import (  # noqa: E402
    EYE_LSTAR_GAP_MIN,
    KITTEN_TEST_IMAGE,
    PREVIEW_JOB_NAME,
    choose_layout,
    focal_face_detail_box,
    is_canvas_job,
    load_saved_labels,
    render_focal_sheet_framed,
    rgb_to_lab,
)
from scipy import ndimage


def _is_kitten_job(meta: dict) -> bool:
    argv = meta.get("argv") or {}
    img = Path(str(argv.get("image", ""))).name
    if img != KITTEN_TEST_IMAGE:
        return False
    w = float(argv.get("wall_width_ft") or meta.get("wall_width_ft") or 0)
    h = float(argv.get("wall_height_ft") or meta.get("wall_height_ft") or 0)
    return is_canvas_job(w, h)


def validate_eye_labels(labels, palette, eye_mask, rgb, min_gap=EYE_LSTAR_GAP_MIN):
    if eye_mask is None or float(np.max(eye_mask)) < 0.25:
        raise AssertionError("kitten eye test: missing eye_mask")
    pal_lab = rgb_to_lab(palette).astype(np.float64)
    em = eye_mask >= 0.45
    cc, n_cc = ndimage.label(em)
    errors = []
    for ci in range(1, n_cc + 1):
        m = cc == ci
        if int(m.sum()) < 30:
            continue
        lids = sorted({int(v) for v in np.unique(labels[m]) if int(v) > 0})
        if len(lids) < 3:
            errors.append(f"eye {ci}: only {len(lids)} label(s) {lids}")
            continue
        lvals = {lid: float(pal_lab[lid, 0]) for lid in lids}
        pupil_lid = min(lids, key=lambda lid: lvals[lid])
        iris_lid = sorted(lids, key=lambda lid: lvals[lid])[min(2, len(lids) - 1)]
        gap = lvals[iris_lid] - lvals[pupil_lid]
        if gap < min_gap - 0.5:
            errors.append(
                f"eye {ci}: L* gap {gap:.1f} between pupil label {pupil_lid + 1} "
                f"and iris label {iris_lid + 1} (need >={min_gap})"
            )
    if errors:
        raise AssertionError("kitten eye test failed:\n  " + "\n  ".join(errors))
    return {"eyes_checked": n_cc, "min_gap_required": min_gap}


def _segment_count_window(gray: np.ndarray, x0: int, y0: int, win: int, dark_cutoff: int = 95) -> int:
    patch = gray[y0: y0 + win, x0: x0 + win]
    if patch.size < win * win:
        return 0
    dark = patch < dark_cutoff
    gy = np.abs(np.diff(patch.astype(np.int16), axis=0))
    gx = np.abs(np.diff(patch.astype(np.int16), axis=1))
    edge = np.zeros_like(patch, dtype=bool)
    edge[1:, :] |= (gy > 32) & dark[1:, :]
    edge[:, 1:] |= (gx > 32) & dark[:, 1:]
    labeled, n = ndimage.label(edge)
    sizes = np.bincount(labeled.ravel())[1:]
    return int((sizes >= 3).sum()) if len(sizes) else 0


def validate_a1_mesh(png_path: Path, window_px: int = 40, max_segments: int = 12, stride: int = 12):
    img = Image.open(png_path).convert("L")
    gray = np.asarray(img)
    worst = (0, 0, 0)
    for y0 in range(0, gray.shape[0] - window_px, stride):
        for x0 in range(0, gray.shape[1] - window_px, stride):
            n = _segment_count_window(gray, x0, y0, window_px)
            if n > worst[0]:
                worst = (n, x0, y0)
    if worst[0] > max_segments:
        raise AssertionError(
            f"kitten mesh test: {png_path.name} has {worst[0]} line segments in "
            f"{window_px}x{window_px} window at ({worst[1]},{worst[2]}); max {max_segments}"
        )
    return {"max_segments_in_window": worst[0], "window_px": window_px, "at": [worst[1], worst[2]]}


def run_from_out_dir(out_dir: Path, render_a1: bool = True, preview_dir: Path | None = None):
    out_dir = Path(out_dir)
    preview_dir = Path(preview_dir) if preview_dir else out_dir
    meta_path = preview_dir / PREVIEW_JOB_NAME
    if not meta_path.is_file():
        meta_path = out_dir / PREVIEW_JOB_NAME
    if not meta_path.is_file():
        raise SystemExit(f"kitten tests: no {PREVIEW_JOB_NAME} under {preview_dir} or {out_dir}")
    meta = json.loads(meta_path.read_text())
    if not _is_kitten_job(meta):
        return {"skipped": True, "reason": "not kitten canvas job"}

    label_dir = meta_path.parent
    labels, palette, regions, face_mask, eye_mask, mouth_mask = load_saved_labels(label_dir)
    argv = meta.get("argv") or {}
    img_path = argv.get("image")
    if not img_path or not Path(img_path).exists():
        raise SystemExit("kitten tests: cannot load source image for L* validation")
    from mural_tiler import apply_image_prep, fit_to_wall
    from PIL import ImageOps

    wall_w_in = float(argv.get("wall_width_ft", meta.get("wall_width_ft", 0))) * 12
    wall_h_in = float(argv.get("wall_height_ft", meta.get("wall_height_ft", 0))) * 12
    ppi = float(meta["ppi"])
    src = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    src = apply_image_prep(src, meta.get("prep") or "none")
    work, _, _ = fit_to_wall(src, wall_w_in, wall_h_in, ppi)
    rgb = np.asarray(work)

    results = {"eye": validate_eye_labels(labels, palette, eye_mask, rgb)}
    if render_a1:
        layout_meta = json.loads((out_dir / "paint_areas.json").read_text()) if (out_dir / "paint_areas.json").is_file() else meta
        scale = float(argv.get("scale", meta.get("scale", layout_meta.get("scale", 3))))
        wall_w_in = float(argv.get("wall_width_ft", meta.get("wall_width_ft", 0))) * 12
        wall_h_in = float(argv.get("wall_height_ft", meta.get("wall_height_ft", 0))) * 12
        overlap = float(argv.get("overlap_in", meta.get("overlap_in", layout_meta.get("sheets", {}).get("overlap_in", 1))))
        mural_w, mural_h = wall_w_in * scale, wall_h_in * scale
        layout = choose_layout(mural_w, mural_h, overlap)
        job = {
            "scale": scale,
            "ppi": ppi,
            "sheet_tint": float(argv.get("sheet_tint", meta.get("sheet_tint", 0))),
        }
        soup = meta.get("soup_clumps") or layout_meta.get("soup_clumps") or []
        box = focal_face_detail_box(soup, eye_mask, mouth_mask, face_mask, labels.shape)
        a1_path = out_dir / "a1_eye_muzzle_300dpi.png"
        pdf_home = out_dir if (out_dir / "mural.pdf").is_file() else job_dir
        rend = render_focal_sheet_framed(
            pdf_home,
            layout,
            job,
            box,
            eye_mask=eye_mask,
            mouth_mask=mouth_mask,
            face_mask=face_mask,
            dpi=300,
        )
        if rend is None:
            raise AssertionError("kitten mesh test: could not render A1 focal sheet at 300 dpi")
        Image.fromarray(rend).save(a1_path)
        results["mesh"] = validate_a1_mesh(a1_path)
        results["a1_path"] = str(a1_path)
    return results


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    prev = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if out is None:
        print("usage: kitten_kit_tests.py <out_dir> [preview_dir]", file=sys.stderr)
        raise SystemExit(2)
    res = run_from_out_dir(out, preview_dir=prev)
    print(json.dumps(res, indent=2))
    if res.get("skipped"):
        raise SystemExit(0)
    raise SystemExit(0)


if __name__ == "__main__":
    main()