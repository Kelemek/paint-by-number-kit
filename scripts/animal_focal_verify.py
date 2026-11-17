#!/usr/bin/env python3
"""Verify animal subject priority: metrics, parity, comparison grid (per image)."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mural_tiler import DETAIL_LEVELS, PREVIEW_JOB_NAME  # noqa: E402
from scripts.rembrandt_focal_verify import SIZES, analyze_dir, build_grid, crop_box, print_row, save_under_mb  # noqa: E402
from scripts.focal_quality import audit_job  # noqa: E402

ANIMALS = [
    ("cat", ROOT / "test_assets" / "cat_pexels.jpg", "cat_pexels"),
    ("dog", ROOT / "test_assets" / "dog_wikimedia.jpg", "dog_wikimedia"),
]


def run_parity(job_dir: Path):
    full = job_dir / "full"
    full.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(ROOT / "mural_tiler.py"), "--from-preview", str(job_dir), "--out", str(full)],
        check=True,
        cwd=ROOT,
    )
    prev = json.loads((job_dir / PREVIEW_JOB_NAME).read_text())["label_metrics"]["regions"]
    areas = json.loads((full / "paint_areas.json").read_text())
    return prev == areas["regions"], areas.get("label_parity", {})


def run_preview(img_path: Path, out: Path, level: str, w: float, h: float, extra=None):
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "mural_tiler.py"),
        "--image",
        str(img_path),
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


def main():
    artifact = Path(os.environ.get("FOCAL_ARTIFACT_DIR", "/opt/cursor/artifacts"))
    artifact.mkdir(parents=True, exist_ok=True)
    after_root = Path(os.environ.get("ANIMAL_AFTER_ROOT", "/tmp/animal_focal_after"))
    baseline_root = Path("/tmp/animal_no_priority_baseline")
    summary_all = []

    for kind, img_path, slug in ANIMALS:
        if not img_path.is_file():
            print(f"missing {img_path}", file=sys.stderr)
            return 1
        for level in DETAIL_LEVELS:
            for name, w, h in SIZES:
                tag = f"{slug}_{level}_{name}"
                out = after_root / tag
                run_preview(img_path, out, level, w, h)
                ok, parity = run_parity(out)
                print(f"{tag} parity_regions={ok} {parity}")
                if name == "wall":
                    aft = analyze_dir(out, "after")
                    print_row(aft)
                    summary_all.append({
                        "animal": kind,
                        "detail": level,
                        "size": name,
                        **{
                            k: aft[k]
                            for k in aft
                            if k not in ("face", "labels", "palette", "prev_rgb", "orig_rgb", "sheet_crop")
                        },
                    })

        base_dir = baseline_root / f"{slug}_medium_wall"
        if not (base_dir / PREVIEW_JOB_NAME).is_file():
            run_preview(
                img_path, base_dir, "medium", 8, 6, extra=["--no-subject-priority"],
            )

        grid_rows = []
        for level in DETAIL_LEVELS:
            wall = after_root / f"{slug}_{level}_wall"
            if not (wall / PREVIEW_JOB_NAME).is_file():
                continue
            aft = analyze_dir(wall, "after")
            bef = analyze_dir(base_dir, "before") if level == "medium" else None
            box = aft["focal_bbox"]
            orig_c = crop_box(aft["orig_rgb"], box)
            bef_c = crop_box(bef["prev_rgb"], box) if bef else None
            aft_c = crop_box(aft["prev_rgb"], box)
            sheet_c = aft["sheet_crop"] if aft.get("sheet_crop") is not None else aft_c
            grid_rows.append((orig_c, bef_c if bef_c is not None else aft_c, aft_c, sheet_c))
        out_img = artifact / f"{slug}_focal_grid.jpg"
        build_grid(grid_rows, out_img)
        for level in DETAIL_LEVELS:
            fails = audit_job(after_root / f"{slug}_{level}_wall", check_sheet_crop=True)
            if fails:
                print(f"QUALITY FAIL {slug}_{level}_wall: {fails}")
        print(f"wrote {out_img}")

    (artifact / "animal_focal_metrics.json").write_text(json.dumps(summary_all, indent=2) + "\n")
    quality_failed = False
    for kind, img_path, slug in ANIMALS:
        for level in DETAIL_LEVELS:
            fails = audit_job(after_root / f"{slug}_{level}_wall", check_sheet_crop=True)
            if fails:
                quality_failed = True
    return 1 if quality_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())