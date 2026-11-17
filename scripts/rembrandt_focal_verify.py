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
    choose_layout,
    fit_to_wall,
    render_focal_sheet_crop,
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


def work_rgb_and_masks(meta, prep, image_path=None):
    src_path = Path(image_path) if image_path else IMG
    src = ImageOps.exif_transpose(Image.open(src_path)).convert("RGB")
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


def skin_mask(face_mask, eye_mask, mouth_mask, shape):
    if face_mask is None:
        return None
    eye = eye_mask >= 0.35 if eye_mask is not None else np.zeros(shape, dtype=bool)
    mouth = mouth_mask >= 0.35 if mouth_mask is not None else np.zeros(shape, dtype=bool)
    return ((face_mask >= 0.35) & ~eye & ~mouth).astype(np.float32)


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
    img_path = Path(meta.get("argv", {}).get("image") or IMG)
    rgb, face = work_rgb_and_masks(meta, prep, img_path)
    scale = float(meta.get("argv", {}).get("scale") or 1.0)
    overlap = float(meta.get("argv", {}).get("overlap_in") or 0.5)
    wall_w_in, wall_h_in = meta["wall_width_ft"] * 12, meta["wall_height_ft"] * 12
    layout = choose_layout(wall_w_in * scale, wall_h_in * scale, overlap)
    job = {"scale": scale, "ppi": meta["ppi"], "sheet_tint": meta.get("argv", {}).get("sheet_tint", 0)}
    data = np.load(job_dir / LABELS_ARCHIVE)
    labels = data["labels"]
    palette = data["palette"]
    regions = data["regions"]
    face_mask = data.get("face_mask")
    eye_mask = data.get("eye_mask")
    mouth_mask = data.get("mouth_mask")
    prev = np.asarray(Image.open(job_dir / "preview.png"))
    sk = skin_mask(face_mask, eye_mask, mouth_mask, labels.shape)
    lm = meta["label_metrics"]
    box = focal_bbox(face)
    sheet = render_focal_sheet_crop(job_dir, layout, job, box, eye_mask, mouth_mask, face_mask)
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
        "original_skin": mask_stats(rgb, sk),
        "preview_skin": preview_stats(prev, labels, palette, sk),
        "focal_bbox": box,
        "sheet_crop": sheet,
        "face": face,
        "labels": labels,
        "palette": palette,
        "prev_rgb": prev,
        "orig_rgb": rgb,
    }
    return row


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
    save_under_mb(grid, out_path)


def save_under_mb(rgb, path: Path, max_bytes=2_900_000):
    im = Image.fromarray(rgb)
    for q in range(92, 40, -8):
        path.parent.mkdir(parents=True, exist_ok=True)
        im.save(path, format="JPEG", quality=q, optimize=True)
        if path.stat().st_size <= max_bytes:
            return
    im = im.resize((im.width * 3 // 4, im.height * 3 // 4), Image.LANCZOS)
    im.save(path, format="JPEG", quality=85, optimize=True)


def print_row(r):
    def fmt(name, orig, prev):
        if not orig or not prev:
            return f"{name}: n/a"
        err = prev["mean_L"] - orig["mean_L"]
        return (
            f"{name}: orig={orig['mean_L']:.1f} preview={prev['mean_L']:.1f} "
            f"error={err:+.1f} (|error|={abs(err):.1f}, tol 5)"
        )

    print(
        f"[{r['tag']}] {r['detail']} {r['wall']} regions={r['regions']} "
        f"face/eye/mouth/bg_regs={r['face_regions']}/{r['eye_regions']}/"
        f"{r['mouth_regions']}/{r['background_regions']}"
    )
    print("  " + fmt("eye", r["original_eye"], r["preview_eye"]))
    print("  " + fmt("mouth", r["original_mouth"], r["preview_mouth"]))
    print("  " + fmt("skin", r.get("original_skin"), r.get("preview_skin")))


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
        sheet_c = aft["sheet_crop"] if aft.get("sheet_crop") is not None else aft_c
        grid_rows.append((orig_c, bef_c if bef_c is not None else aft_c, aft_c, sheet_c))

    build_grid(grid_rows, artifact / "rembrandt_focal_grid.jpg")
    med = after_root / "medium_wall" / "preview.png"
    if med.is_file():
        save_under_mb(np.asarray(Image.open(med)), artifact / "rembrandt_after_preview_medium.jpg")

    report = artifact / "rembrandt_focal_metrics.json"
    summary = []
    for level in LEVELS:
        for name, _, _ in SIZES:
            tag = f"{level}_{name}"
            p = after_root / tag
            if (p / PREVIEW_JOB_NAME).is_file():
                r = analyze_dir(p, "after")
                summary.append({
                    k: r[k]
                    for k in r
                    if k not in ("face", "labels", "palette", "prev_rgb", "orig_rgb", "sheet_crop")
                })
    report.write_text(json.dumps(summary, indent=2) + "\n")
    from scripts.focal_quality import audit_job

    quality_failed = False
    for level in LEVELS:
        tag = f"{level}_wall"
        fails = audit_job(after_root / tag, check_sheet_crop=True)
        if fails:
            quality_failed = True
            print(f"QUALITY FAIL {tag}: {fails}")
    print(f"wrote {artifact / 'rembrandt_focal_grid.jpg'}")
    print(f"wrote {report}")
    return 1 if quality_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())