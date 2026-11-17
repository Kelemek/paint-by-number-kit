#!/usr/bin/env python3
"""Legibility verification: focal grids at 300 dpi, metrics, before/after comparison."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mural_tiler import DETAIL_LEVELS, PREVIEW_JOB_NAME  # noqa: E402
from scripts.focal_quality import audit_job, audit_legibility  # noqa: E402
from scripts.rembrandt_focal_verify import SIZES, analyze_dir, build_grid, crop_box, print_row, save_under_mb  # noqa: E402

CASES = [
    ("kitten", ROOT / "test_assets" / "kitten_ragdoll_canvas.jpg", "kitten", "", (8 / 12, 10 / 12)),
    ("rembrandt", ROOT / "test_assets" / "rembrandt_1634.jpg", "rembrandt", "", (8, 6)),
    ("cat", ROOT / "test_assets" / "cat_pexels.jpg", "cat_pexels", "", (8, 6)),
    ("dog", ROOT / "test_assets" / "dog_wikimedia.jpg", "dog_wikimedia", "", (8, 6)),
]


def run_preview(img: Path, out: Path, level: str, w: float, h: float, extra=None):
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "mural_tiler.py"),
        "--image",
        str(img),
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
    if extra:
        cmd.extend(extra)
    subprocess.run(cmd, check=True, cwd=ROOT)


def run_full(preview_dir: Path, detail_inset: str = ""):
    full = preview_dir / "full"
    full.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "mural_tiler.py"), "--from-preview", str(preview_dir), "--out", str(full)]
    if detail_inset:
        cmd.extend(["--detail-inset", detail_inset])
    subprocess.run(cmd, check=True, cwd=ROOT)


def legibility_crop(row, dpi=300):
    """300 dpi sheet tile crop for the eye/nose focal box."""
    return row.get("sheet_crop")


def main():
    artifact = Path(os.environ.get("LEGIBILITY_ARTIFACT_DIR", "/opt/cursor/artifacts"))
    artifact.mkdir(parents=True, exist_ok=True)
    after_root = Path(os.environ.get("LEGIBILITY_AFTER_ROOT", "/tmp/legibility_after"))
    before_root = Path(os.environ.get("LEGIBILITY_BEFORE_ROOT", "/tmp/legibility_before"))
    summary = []
    failed = False

    for slug, img_path, name, subject_box, (wall_w, wall_h) in CASES:
        if not img_path.is_file():
            print(f"missing {img_path}", file=sys.stderr)
            return 1
        extra = ["--subject-box", subject_box] if subject_box else None
        is_canvas = slug == "kitten"
        w_run, h_run = ((8 / 12, 10 / 12) if is_canvas else (wall_w, wall_h))
        levels = list(DETAIL_LEVELS) if is_canvas else ["medium"]
        for level in levels:
            tag = f"{name}_{level}_{'canvas' if is_canvas else 'wall'}"
            out = after_root / tag
            soup = ["--soup-report"] if is_canvas and level == "medium" else []
            run_preview(img_path, out, level, w_run, h_run, extra=(extra or []) + soup)
            inset = "auto" if is_canvas and level == "medium" else ""
            run_full(out, detail_inset=inset)

        grid_rows = []
        for level in levels:
            tag = f"{name}_{level}_{'canvas' if is_canvas else 'wall'}"
            aft = analyze_dir(after_root / tag, "after")
            print_row(aft)
            bef = None
            bef_dir = before_root / tag
            if (bef_dir / PREVIEW_JOB_NAME).is_file():
                bef = analyze_dir(bef_dir, "before")
            box = aft["focal_bbox"]
            orig_c = crop_box(aft["orig_rgb"], box)
            bef_c = crop_box(bef["prev_rgb"], box) if bef else None
            aft_c = crop_box(aft["prev_rgb"], box)
            sheet_c = aft.get("sheet_crop")
            if sheet_c is None:
                sheet_c = aft_c
            grid_rows.append((orig_c, bef_c if bef_c is not None else aft_c, aft_c, sheet_c))

            leg = audit_legibility(after_root / tag)
            fq = audit_job(after_root / tag, check_sheet_crop=True, check_legibility=True)
            summary.append(
                {
                    "case": tag,
                    "regions": aft["regions"],
                    "colors": len(aft["palette"]),
                    "legibility": leg,
                    "focal_quality": fq,
                    "original_eye": aft.get("original_eye"),
                    "preview_eye": aft.get("preview_eye"),
                }
            )
            if leg or fq:
                failed = True
                print(f"FAIL {tag}: legibility={leg} focal={fq}")

        grid_path = artifact / f"{name}_legibility_grid.jpg"
        build_grid(grid_rows, grid_path)
        print(f"wrote {grid_path}")
        if is_canvas and (after_root / f"{name}_medium_canvas" / "preview.png").is_file():
            save_under_mb(
                np.asarray(Image.open(after_root / f"{name}_medium_canvas" / "preview.png")),
                artifact / "kitten_after_preview_medium.jpg",
            )

    report = artifact / "legibility_metrics.json"
    report.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {report}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())