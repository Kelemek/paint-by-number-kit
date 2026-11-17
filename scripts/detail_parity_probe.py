#!/usr/bin/env python3
"""Compare preview vs full-build parity across skill detail levels."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG = Path(os.environ.get("PARITY_IMAGE", ROOT / "test_assets" / "preview_face.png"))
if "PARITY_FACE_BOX" in os.environ:
    FACE = os.environ["PARITY_FACE_BOX"]
elif Path(os.environ.get("PARITY_IMAGE", ROOT / "test_assets/preview_face.png")).name == "preview_face.png":
    FACE = "120,80,360,480"
else:
    FACE = ""


def run(cmd):
    subprocess.run(cmd, check=True, cwd=ROOT)


def metrics(job_dir, preview_dir=None):
    areas = json.loads((job_dir / "paint_areas.json").read_text())
    preview_dir = preview_dir or job_dir
    prev = json.loads((preview_dir / "preview_job.json").read_text())
    return {
        "regions": areas["regions"],
        "preview_regions": prev["label_metrics"]["regions"],
        "detail": areas.get("detail"),
        "unlabeled": areas.get("unlabeled_regions"),
        "face_regions": areas.get("label_metrics", {}).get("face_regions"),
        "parity": areas.get("label_parity"),
        "trace_sigma": areas.get("trace_sigma"),
    }


def main():
    if not IMG.is_file():
        print(f"missing {IMG}", file=sys.stderr)
        return 1
    levels = ["low", "medium", "high", "photo"]
    sizes = [("wall", 8, 6), ("canvas", 2, 1.5)]
    print("level,size,preview_regions,full_regions,unlabeled,face_regions,parity,trace_sigma,monotonic_ok")
    failed_mono = False
    prev_by_size = {name: 0 for name, _, _ in sizes}
    for level in levels:
        for name, w, h in sizes:
            base = Path(f"/tmp/parity_{name}_{level}")
            base.mkdir(parents=True, exist_ok=True)
            cmd = [
                "python3", "mural_tiler.py",
                "--image", str(IMG),
                "--wall-width-ft", str(w), "--wall-height-ft", str(h),
                "--detail", level, "--preview-only", "--out", str(base),
            ]
            if FACE:
                cmd.extend(["--face-box", FACE])
            run(cmd)
            run(["python3", "mural_tiler.py", "--from-preview", str(base), "--out", str(base / "full")])
            prev = metrics(base)
            full = metrics(base / "full", base)
            ok = full["parity"] and full["parity"].get("regions_match")
            mono = full["regions"] >= prev_by_size[name]
            if not mono:
                failed_mono = True
                print(f"  MONOTONIC FAIL: {level} {name} has {full['regions']} regions < {prev_by_size[name]}")
            prev_by_size[name] = max(prev_by_size[name], full["regions"])
            print(
                f"{level},{name},{prev['regions']},{full['regions']},{full['unlabeled']},"
                f"{full['face_regions']},{ok},{full['trace_sigma']},{mono}"
            )
            # Manual mismatch probe: rerun without detail/prep/face-box (simulates organizer typo)
            mismatch = Path(f"/tmp/parity_{name}_{level}_mismatch")
            mismatch.mkdir(parents=True, exist_ok=True)
            run([
                "python3", "mural_tiler.py",
                "--image", str(IMG),
                "--wall-width-ft", str(w), "--wall-height-ft", str(h),
                "--colors", "8", "--no-face-priority",
                "--out", str(mismatch),
            ])
            bad = json.loads((mismatch / "paint_areas.json").read_text())["regions"]
            print(f"  mismatch_rerun_regions={bad} (preview had {prev['regions']})")
            if FACE and name == "wall":
                from scripts.focal_quality import audit_job

                fq = audit_job(base, check_sheet_crop=True)
                if fq:
                    failed_mono = True
                    print(f"  FOCAL_QUALITY FAIL: {fq}")
    return 1 if failed_mono else 0


if __name__ == "__main__":
    raise SystemExit(main())