#!/usr/bin/env python3
"""Fail when mural sheets place numbers under 6 pt or with overlapping boxes."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from pypdf import PdfReader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mural_tiler import (  # noqa: E402
    LABELS_ARCHIVE,
    MIN_LEGIBLE_PT,
    PREVIEW_JOB_NAME,
    STYLES,
    _aabb_overlap,
    _number_aabb_work,
    choose_layout,
    collect_placed_numbers,
    load_saved_labels,
)


def check_placement(job_dir: Path) -> list[str]:
    meta = json.loads((job_dir / PREVIEW_JOB_NAME).read_text())
    labels, palette, regions, face_mask, eye_mask, mouth_mask = load_saved_labels(job_dir)
    style = STYLES[meta.get("transfer", "trace")]
    job = {
        "wall_w_in": meta["wall_width_ft"] * 12,
        "wall_h_in": meta["wall_height_ft"] * 12,
        "scale": meta.get("scale", 1.0),
        "ppi": meta["ppi"],
    }
    mural_w, mural_h = job["wall_w_in"] * job["scale"], job["wall_h_in"] * job["scale"]
    layout = choose_layout(mural_w, mural_h, meta.get("overlap_in", 0.25))
    placed, _, _ = collect_placed_numbers(
        labels, regions, layout, job, style, face_mask, eye_mask, mouth_mask,
    )
    px_to_pt = (job["scale"] / job["ppi"]) * 72.0
    errs = []
    boxes = []
    for x, y, text, size in placed:
        if size < MIN_LEGIBLE_PT - 1e-6:
            errs.append(f"number {text} at ({x:.1f},{y:.1f}) is {size:.2f} pt (< {MIN_LEGIBLE_PT} pt)")
        box = _number_aabb_work(x, y, text, size, style.font, px_to_pt)
        for other in boxes:
            if _aabb_overlap(box, other):
                errs.append(f"numbers {text} overlaps another at ({x:.1f},{y:.1f})")
                break
        boxes.append(box)
    return errs


def rasterize_pdf_page(pdf_path: Path, page: int, dpi: int, out_png: Path) -> bool:
    try:
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                str(page),
                "-l",
                str(page),
                "-png",
                "-r",
                str(dpi),
                "-singlefile",
                str(pdf_path),
                str(out_png.with_suffix("")),
            ],
            check=True,
            capture_output=True,
        )
        return out_png.is_file()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("job_dir", type=Path, help="preview job directory (preview_job.json + labels.npz)")
    ap.add_argument("--dpi", type=int, default=300, help="raster DPI for optional PDF sanity check")
    args = ap.parse_args()
    job_dir = args.job_dir.resolve()
    errs = check_placement(job_dir)
    mural = job_dir / "mural.pdf"
    if not mural.is_file():
        full = job_dir / "full" / "mural.pdf"
        if full.is_file():
            mural = full
    if mural.is_file():
        try:
            n_pages = len(PdfReader(str(mural)).pages)
            if n_pages > 0:
                tmp = job_dir / ".legibility_raster.png"
                rasterize_pdf_page(mural, 1, args.dpi, tmp)
        except Exception as exc:
            errs.append(f"pdf raster skipped: {exc}")
    if errs:
        print("\n".join(errs), file=sys.stderr)
        raise SystemExit(1)
    meta = json.loads((job_dir / PREVIEW_JOB_NAME).read_text())
    data = np.load(job_dir / LABELS_ARCHIVE)
    print(
        f"ok: {job_dir.name} regions={int(data['regions'].max())} "
        f"wall={meta['wall_width_ft']}x{meta['wall_height_ft']} ft detail={meta.get('detail')}"
    )


if __name__ == "__main__":
    main()