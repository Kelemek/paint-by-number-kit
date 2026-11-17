#!/usr/bin/env python3
"""Fail when mural sheets place numbers under 6 pt or with overlapping boxes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pypdf import PdfReader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from legibility_layout import (  # noqa: E402
    MIN_LEGIBLE_PT,
    _aabb_overlap,
    _number_aabb_work,
    _segment_intersects_aabb,
)
from mural_tiler import (  # noqa: E402
    LABELS_ARCHIVE,
    PREVIEW_JOB_NAME,
    STYLES,
    choose_layout,
    collect_placed_numbers,
    load_saved_labels,
    merge_crowded_regions,
    label_regions,
)


def check_placement(job_dir: Path) -> list[str]:
    meta = json.loads((job_dir / PREVIEW_JOB_NAME).read_text())
    labels, palette, _r0, face_mask, eye_mask, mouth_mask = load_saved_labels(job_dir)
    face = meta.get("subject_priority") or meta.get("face_priority") or {}
    style = STYLES[meta.get("transfer", "trace")]
    job = {
        "wall_w_in": meta["wall_width_ft"] * 12,
        "wall_h_in": meta["wall_height_ft"] * 12,
        "scale": meta.get("scale", 1.0),
        "ppi": meta["ppi"],
    }
    labels, regions, _, _ = merge_crowded_regions(
        labels,
        palette,
        job["ppi"],
        job["scale"],
        style,
        eye_mask=eye_mask,
        mouth_mask=mouth_mask,
        eye_disks_work=face.get("eye_disks_work"),
    )
    mural_w, mural_h = job["wall_w_in"] * job["scale"], job["wall_h_in"] * job["scale"]
    layout = choose_layout(mural_w, mural_h, meta.get("overlap_in", 0.25))
    placements, labeled_all, guide_all, inside_all, callout_all = collect_placed_numbers(
        labels, regions, layout, job, style, face_mask, eye_mask, mouth_mask,
    )
    px_to_pt = (job["scale"] / job["ppi"]) * 72.0
    errs = []
    boxes = []
    for p in placements:
        if p.size < MIN_LEGIBLE_PT - 1e-6:
            errs.append(
                f"number {p.text} for region {p.region_id} is {p.size:.2f} pt (< {MIN_LEGIBLE_PT} pt)"
            )
        box = _number_aabb_work(p.num_x, p.num_y, p.text, p.size, style.font, px_to_pt)
        for other in boxes:
            if _aabb_overlap(box, other):
                errs.append(f"number {p.text} overlaps another label")
                break
        if p.is_callout:
            for other in boxes:
                if _segment_intersects_aabb(p.dot_x, p.dot_y, p.num_x, p.num_y, other):
                    errs.append(f"leader for {p.text} crosses another number box")
                    break
            if p.region_id not in labeled_all:
                errs.append(f"callout region {p.region_id} not counted as labeled")
        boxes.append(box)
    for rid in callout_all | inside_all:
        if rid not in labeled_all:
            errs.append(f"region {rid} marked placed but missing from labeled set")
    return errs


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
                pass
        except Exception as exc:
            errs.append(f"pdf read skipped: {exc}")
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