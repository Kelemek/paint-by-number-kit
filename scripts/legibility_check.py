#!/usr/bin/env python3
"""Fail when mural sheets place numbers under 6 pt, leaders over 12 mm, or overlapping boxes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from pypdf import PdfReader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from legibility_layout import (  # noqa: E402
    MAX_LEADER_MM,
    MIN_LEGIBLE_PT,
    analyze_placement_layout,
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
)


def evaluate_job(job_dir: Path) -> tuple[list[str], dict]:
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
    stats = analyze_placement_layout(placements, px_to_pt, style, job["ppi"], job["scale"])
    stats["numbered_inside"] = len(inside_all)
    stats["callout_regions"] = len(callout_all)
    stats["guide_regions"] = len(guide_all)
    stats["regions_after_merge"] = int(regions.max())

    errs = []
    boxes = []
    mm_per_px = 25.4 / (job["ppi"] * job["scale"])
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
            length_mm = math.hypot(p.num_x - p.dot_x, p.num_y - p.dot_y) * mm_per_px
            if length_mm > MAX_LEADER_MM + 1e-6:
                errs.append(f"leader for {p.text} is {length_mm:.2f} mm (> {MAX_LEADER_MM} mm)")
            for other in boxes:
                if _segment_intersects_aabb(p.dot_x, p.dot_y, p.num_x, p.num_y, other):
                    errs.append(f"leader for {p.text} crosses another number box")
                    break
        boxes.append(box)
    for rid in callout_all | inside_all:
        if rid not in labeled_all:
            errs.append(f"region {rid} marked placed but missing from labeled set")
    return errs, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("job_dir", type=Path, help="preview job directory (preview_job.json + labels.npz)")
    ap.add_argument("--stats-json", type=Path, help="write layout statistics JSON")
    ap.add_argument("--compare-json", type=Path, help="optional prior stats JSON for before/after")
    args = ap.parse_args()
    job_dir = args.job_dir.resolve()
    errs, stats = evaluate_job(job_dir)
    if args.compare_json and args.compare_json.is_file():
        prior = json.loads(args.compare_json.read_text())
        stats["step2_reference"] = prior
        stats["vs_step2"] = {
            "leader_max_mm_delta": round(stats["leader_max_mm"] - prior.get("leader_max_mm", 0), 3),
            "leader_median_mm_delta": round(
                stats["leader_median_mm"] - prior.get("leader_median_mm", 0), 3
            ),
            "crossings_delta": stats["leader_line_crossings"] - prior.get("leader_line_crossings", 0),
            "min_number_gap_mm_delta": round(
                stats["min_number_gap_mm"] - prior.get("min_number_gap_mm", 0), 3
            ),
        }
    if args.stats_json:
        args.stats_json.write_text(json.dumps(stats, indent=2) + "\n")
    mural = job_dir / "mural.pdf"
    if mural.is_file():
        try:
            PdfReader(str(mural))
        except Exception as exc:
            errs.append(f"pdf read skipped: {exc}")
    if errs:
        print("\n".join(errs), file=sys.stderr)
        print(json.dumps(stats, indent=2))
        raise SystemExit(1)
    print(
        f"ok: leaders max={stats['leader_max_mm']} mm median={stats['leader_median_mm']} mm "
        f"crossings={stats['leader_line_crossings']} min_gap={stats['min_number_gap_mm']} mm"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()