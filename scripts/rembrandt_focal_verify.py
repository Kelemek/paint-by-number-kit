#!/usr/bin/env python3
"""Rembrandt focal verification: L* in eyes/mouth, region counts, comparison grid."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mural_tiler import (  # noqa: E402
    DETAIL_LEVELS,
    PREVIEW_JOB_NAME,
    LABELS_ARCHIVE,
    apply_image_prep,
    fit_to_wall,
    rgb_to_lab,
    resolve_faces,
)

IMG = ROOT / "test_assets" / "rembrandt_1634.jpg"
LEVELS = sorted(DETAIL_LEVELS)
SIZES = [("wall", 8, 6), ("canvas", 2, 1.5)]


def run_preview(out: Path, level: str, w: float, h: float):
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "mural_tiler.py"),
        "--image",
        str(IMG),
        "--wall-width-ft",
        str(w),
        "--wall-height-ft",
        str(h),
        "--detail",
        level,
        "--preview-only",
        "--preview-sheets",
        "2",
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


def work_rgb_and_masks(meta, prep):
    src = ImageOps.exif_transpose(Image.open(IMG)).convert("RGB")
    src = apply_image_prep(src, prep)
    wall_w_in, wall_h_in = meta["wall_width_ft"] * 12, meta["wall_height_ft"] * 12
    ppi = float(meta["ppi"])
    img, _, crop_box = fit_to_wall(src, wall_w_in, wall_h_in, ppi)
    rgb = np.asarray(img)
    class A:
        pass

    a = A()
    a.no_face_priority = False
    a.face_box = ""
    a.colors = meta["colors"]
    a.detail = meta["detail"]
    a.min_region_in = meta["min_region_in"]
    face = resolve_faces(src, img, crop_box, a)
    return rgb, face


def mask_stats(rgb_u8, mask, thresh=0.35):
    m = mask >= thresh if mask is not None else np.zeros(rgb_u8.shape[:2], dtype=bool)
    if not np.any(m):
        return None
    lab = rgb_to_lab(rgb_u8[m])
    return {
        "mean_L": round(float(lab[:, 0].mean()), 2),
        "min_L": round(float(lab[:, 0].min()), 2),
        "pixels": int(m.sum()),
    }


def preview_stats(preview_rgb, labels, palette, mask, thresh=0.35):
    m = mask >= thresh if mask is not None else np.zeros(labels.shape, dtype=bool)
    if not np.any(m):
        return None
    px = palette[labels[m]]
    lab = rgb_to_lab(px)
    dark_idx = int(np.argmin(rgb_to_lab(palette)[:, 0]))
    dark_share = float((labels[m] == dark_idx).mean())
    return {
        "mean_L": round(float(lab[:, 0].mean()), 2),
        "min_L": round(float(lab[:, 0].min()), 2),
        "darkest_palette_share": round(dark_share, 4),
        "darkest_palette_hex": "#%02x%02x%02x" % tuple(int(v) for v in palette[dark_idx]),
    }


def focal_bbox(face, pad=0.15):
    pts = []
    for d in face.get("eye_disks_work") or []:
        cx, cy, r = d
        pts.extend([(cx - r, cy - r), (cx + r, cy + r)])
    for e in face.get("mouth_ellipses_work") or []:
        cx, cy, rx, ry = e
        pts.extend([(cx - rx, cy - ry), (cx + rx, cy + ry)])
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return int(max(0, x0 - dx)), int(max(0, y0 - dy)), int(x1 + dx), int(y1 + dy)


def crop_box(arr, box):
    if box is None:
        return arr
    x0, y0, x1, y1 = box
    return arr[y0:y1, x0:x1]


def analyze_dir(job_dir: Path, tag: str):
    meta = json.loads((job_dir / PREVIEW_JOB_NAME).read_text())
    prep = meta.get("prep") or "auto"
    rgb, face = work_rgb_and_masks(meta, prep)
    data = np.load(job_dir / LABELS_ARCHIVE)
    labels = data["labels"]
    palette = data["palette"]
    regions = data["regions"]
    face_mask = data.get("face_mask")
    eye_mask = data.get("eye_mask")
    mouth_mask = data.get("mouth_mask")
    prev = np.asarray(Image.open(job_dir / "preview.png"))
    lm = meta["label_metrics"]
    row = {
        "tag": tag,
        "detail": meta["detail"],
        "wall": f"{meta['argv']['wall_width_ft']}x{meta['argv']['wall_height_ft']}",
        "regions": lm["regions"],
        "face_regions": lm.get("face_regions"),
        "eye_regions": lm.get("eye_regions"),
        "mouth_regions": lm.get("mouth_regions"),
        "background_regions": lm.get("background_regions"),
        "original_eye": mask_stats(rgb, eye_mask),
        "original_mouth": mask_stats(rgb, mouth_mask),
        "preview_eye": preview_stats(prev, labels, palette, eye_mask),
        "preview_mouth": preview_stats(prev, labels, palette, mouth_mask),
        "focal_bbox": focal_bbox(face),
        "face": face,
        "labels": labels,
        "palette": palette,
        "prev_rgb": prev,
        "orig_rgb": rgb,
    }
    return row


def load_sheet_page_crop(job_dir: Path, box):
    pages = sorted((job_dir / "preview_pages").glob("page-*.png"))
    if not pages or box is None:
        return None
    # Sample sheet is usually last pages; pick largest crop overlap with focal box via heuristics: page 4+ often sheets
    for p in reversed(pages):
        im = np.asarray(Image.open(p).convert("RGB"))
        # Sheet pages are landscape letter; use center band as rough mural tile for wall jobs
        h, w = im.shape[:2]
        tile = im[int(h * 0.12) : int(h * 0.88), int(w * 0.08) : int(w * 0.92)]
        x0, y0, x1, y1 = box
        fw, fh = x1 - x0, y1 - y0
        crop = np.asarray(
            Image.fromarray(tile).resize((fw, fh), Image.LANCZOS)
        )
        if crop.size:
            return crop
    return None


def build_grid(rows_by_level, out_path: Path):
    """rows: list of (orig, before, after, after_sheet) uint8 crops."""
    if not rows_by_level:
        return
    cell_h = max(r[0].shape[0] for r in rows_by_level)
    cell_w = max(r[0].shape[1] for r in rows_by_level)
    ncols = 4
    labels_h = 28
    canvas_h = len(rows_by_level) * (cell_h + labels_h) + labels_h
    canvas_w = ncols * cell_w
    grid = np.full((canvas_h, canvas_w, 3), 245, np.uint8)
    titles = ["original crop", "before preview", "after preview", "after PDF sheet"]
    for i, t in enumerate(titles):
        pass  # optional title strip skipped for brevity
    for ri, (orig, bef, aft, sheet) in enumerate(rows_by_level):
        y0 = labels_h + ri * (cell_h + labels_h)
        for ci, tile in enumerate((orig, bef, aft, sheet if sheet is not None else aft)):
            if tile is None:
                continue
            im = Image.fromarray(tile).resize((cell_w, cell_h), Image.LANCZOS)
            grid[y0 : y0 + cell_h, ci * cell_w : (ci + 1) * cell_w] = np.asarray(im)
    Image.fromarray(grid).save(out_path, optimize=True)


def print_row(r):
    def fmt(name, orig, prev):
        if not orig or not prev:
            return f"{name}: n/a"
        d_mean = orig["mean_L"] - prev["mean_L"]
        d_min = orig["min_L"] - prev["min_L"]
        extra = f", darkest_share={prev.get('darkest_palette_share')}" if "darkest_palette_share" in prev else ""
        return (
            f"{name}: orig L* mean/min={orig['mean_L']}/{orig['min_L']} "
            f"preview {prev['mean_L']}/{prev['min_L']} (Δmean={d_mean:+.1f}, Δmin={d_min:+.1f}{extra})"
        )

    print(
        f"[{r['tag']}] {r['detail']} {r['wall']} regions={r['regions']} "
        f"face/eye/mouth/bg_regs={r['face_regions']}/{r['eye_regions']}/"
        f"{r['mouth_regions']}/{r['background_regions']}"
    )
    print("  " + fmt("eye", r["original_eye"], r["preview_eye"]))
    print("  " + fmt("mouth", r["original_mouth"], r["preview_mouth"]))


def main():
    artifact = Path(os.environ.get("FOCAL_ARTIFACT_DIR", "/opt/cursor/artifacts"))
    artifact.mkdir(parents=True, exist_ok=True)
    before_root = Path(os.environ.get("FOCAL_BEFORE_ROOT", "/tmp/rembrandt_focal_before"))
    after_root = Path(os.environ.get("FOCAL_AFTER_ROOT", "/tmp/rembrandt_focal_after"))

    for level in LEVELS:
        for name, w, h in SIZES:
            tag = f"{level}_{name}"
            run_preview(after_root / tag, level, w, h)

    grid_rows = []
    for level in LEVELS:
        tag = f"{level}_wall"
        aft = analyze_dir(after_root / tag, "after")
        print_row(aft)
        bef = None
        bef_dir = before_root / tag
        if (bef_dir / PREVIEW_JOB_NAME).is_file():
            bef = analyze_dir(bef_dir, "before")
            print_row(bef)
        box = aft["focal_bbox"]
        orig_c = crop_box(aft["orig_rgb"], box)
        bef_c = crop_box(bef["prev_rgb"], box) if bef else None
        aft_c = crop_box(aft["prev_rgb"], box)
        sheet_c = load_sheet_page_crop(after_root / tag, box)
        grid_rows.append((orig_c, bef_c if bef_c is not None else aft_c, aft_c, sheet_c))

    build_grid(grid_rows, artifact / "rembrandt_focal_grid.png")
    med = after_root / "medium_wall" / "preview.png"
    if med.is_file():
        Image.open(med).save(artifact / "rembrandt_after_preview_medium.png", optimize=True)

    report = artifact / "rembrandt_focal_metrics.json"
    summary = []
    for level in LEVELS:
        for name, _, _ in SIZES:
            tag = f"{level}_{name}"
            p = after_root / tag
            if (p / PREVIEW_JOB_NAME).is_file():
                r = analyze_dir(p, "after")
                summary.append({k: r[k] for k in r if k not in ("face", "labels", "palette", "prev_rgb", "orig_rgb")})
    report.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {artifact / 'rembrandt_focal_grid.png'}")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())