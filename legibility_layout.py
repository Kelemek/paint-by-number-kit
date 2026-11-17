"""Merge crowded micro-regions and place leader-line callouts for legible numbering."""
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from reportlab.pdfbase.pdfmetrics import stringWidth
from scipy import ndimage

PT = 72.0
MIN_LEGIBLE_PT = 6.0
PAINTABLE_DIAM_MM = 2.5
MAX_LEADER_MM = 15.0
LEADER_LINE_PT = 0.3
LEADER_GRAY = 0.35
EYE_SPOTS_PER_DISK = 8
MOUTH_PROTECTED_SPOTS = 4


@dataclass(frozen=True)
class NumberPlacement:
    region_id: int
    text: str
    size: float
    num_x: float
    num_y: float
    dot_x: float | None = None
    dot_y: float | None = None

    @property
    def is_callout(self):
        return self.dot_x is not None

    @property
    def is_inside(self):
        return self.dot_x is None


def _rgb_to_lab(rgb):
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def _lab_dist(palette, ci, cj):
    a = _rgb_to_lab(np.asarray(palette[ci], dtype=np.uint8).reshape(1, 3))[0]
    b = _rgb_to_lab(np.asarray(palette[cj], dtype=np.uint8).reshape(1, 3))[0]
    return float(np.linalg.norm(a - b))


def paintable_radius_work_px(ppi, scale=1.0):
    diam_in = (PAINTABLE_DIAM_MM / 25.4) * scale
    return diam_in * ppi / 2.0


def max_leader_work_px(ppi, scale=1.0):
    return (MAX_LEADER_MM / 25.4) * scale * ppi


def region_inscribed_radii(labels, regions):
    dist = interior_distance(labels)
    ids = np.arange(1, int(regions.max()) + 1)
    if len(ids) == 0:
        return {}
    radii = ndimage.maximum(dist, regions, ids)
    return {int(rid): float(radii[rid - 1]) for rid in ids}


def interior_distance(labels):
    p = np.pad(labels, 1, constant_values=-1)
    c = p[1:-1, 1:-1]
    edge = (c != p[:-2, 1:-1]) | (c != p[2:, 1:-1]) | (c != p[1:-1, :-2]) | (c != p[1:-1, 2:])
    return ndimage.distance_transform_edt(~edge)


def label_regions(labels):
    regions = np.zeros(labels.shape, dtype=np.int32)
    colors = [-1]
    for k in np.unique(labels):
        lab, count = ndimage.label(labels == k)
        regions[lab > 0] = lab[lab > 0] + len(colors) - 1
        colors += [int(k)] * count
    return regions, np.array(colors)


def max_fit_number_pt(radius, text, px_to_pt, style):
    if radius <= 0:
        return 0.0
    half_diag = math.hypot(stringWidth(text, style.font, 1), 0.72) / 2
    return min(style.max_pt, 0.85 * radius * px_to_pt / half_diag)


def legible_number_pt(radius, text, px_to_pt, style):
    fit = max_fit_number_pt(radius, text, px_to_pt, style)
    if fit < MIN_LEGIBLE_PT:
        return None
    return min(style.max_pt, max(MIN_LEGIBLE_PT, fit))


def _number_aabb_work(x, y, text, size, font, px_to_pt):
    hw = stringWidth(text, font, size) / 2 / px_to_pt
    hh = (0.36 * size) / px_to_pt
    return x - hw, y - hh, x + hw, y + hh


def _aabb_overlap(a, b, pad=0.35):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def _segment_intersects_aabb(x0, y0, x1, y1, box, pad=0.2):
    """Rough segment vs axis-aligned box test."""
    bx0, by0, bx1, by1 = box
    bx0 -= pad
    by0 -= pad
    bx1 += pad
    by1 += pad
    if max(x0, x1) < bx0 or min(x0, x1) > bx1 or max(y0, y1) < by0 or min(y0, y1) > by1:
        return False
    # Liang-Barsky lite: sample along segment
    steps = max(4, int(math.hypot(x1 - x0, y1 - y0) * 2))
    for t in np.linspace(0, 1, steps):
        px = x0 + t * (x1 - x0)
        py = y0 + t * (y1 - y0)
        if bx0 <= px <= bx1 and by0 <= py <= by1:
            return True
    return False


def _eye_disk_region_ids(regions, eye_mask, cx, cy, r, palette, labels):
    """Keep a small set of distinct eye-feature regions inside one eye disk."""
    h, w = regions.shape
    yy, xx = np.mgrid[0:h, 0:w]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    if not np.any(disk):
        return set()
    rids = [int(rid) for rid in np.unique(regions[disk]) if rid > 0]
    if not rids:
        return set()
    areas = {rid: int((regions == rid).sum()) for rid in rids}
    ranked = sorted(rids, key=lambda rid: areas[rid], reverse=True)
    kept = []
    pal_lab = None
    for rid in ranked:
        ci = int(labels[regions == rid][0])
        if len(kept) >= EYE_SPOTS_PER_DISK:
            break
        if not kept:
            kept.append(rid)
            continue
        if pal_lab is None:
            pal_lab = _rgb_to_lab(palette)
        distinct = all(abs(pal_lab[ci, 0] - pal_lab[int(labels[regions == k][0]), 0]) > 4.0 for k in kept)
        if distinct or areas[rid] >= areas[kept[-1]] * 0.35:
            kept.append(rid)
    return set(kept)


def protected_feature_region_ids(regions, labels, palette, eye_mask, mouth_mask, eye_disks_work=None):
    protected = set()
    if eye_disks_work:
        for cx, cy, r in eye_disks_work:
            protected |= _eye_disk_region_ids(regions, eye_mask, cx, cy, r, palette, labels)
    if mouth_mask is not None and float(np.max(mouth_mask)) > 0.1:
        m = mouth_mask >= 0.35
        cand = []
        for rid in np.unique(regions[m]):
            if rid <= 0:
                continue
            area = int((regions == rid).sum())
            ci = int(labels[regions == rid][0])
            cand.append((area, ci, int(rid)))
        cand.sort(reverse=True)
        pal_l = _rgb_to_lab(palette)[:, 0]
        picked = []
        for area, ci, rid in cand:
            if len(picked) >= MOUTH_PROTECTED_SPOTS:
                break
            if not picked or min(abs(pal_l[ci] - pal_l[p[1]]) for p in picked) > 3.0:
                picked.append((area, ci, rid))
        protected |= {rid for _, _, rid in picked}
    return protected


def _region_tiny_for_merge(radius, color_idx, px_to_pt, style, paintable_r_px):
    text = str(color_idx + 1)
    if legible_number_pt(radius, text, px_to_pt, style) is not None:
        return False
    return radius < paintable_r_px


def merge_crowded_regions(
    labels,
    palette,
    ppi,
    scale,
    style,
    eye_mask=None,
    mouth_mask=None,
    eye_disks_work=None,
    max_passes=2000,
):
    """Merge sub-paintable regions into the closest-colored neighbor."""
    px_to_pt = (scale / ppi) * PT
    paintable_r = paintable_radius_work_px(ppi, scale)
    pre_regions, _ = label_regions(labels)
    pre_count = int(pre_regions.max())
    merged = 0
    for _ in range(max_passes):
        regions, colors = label_regions(labels)
        radii = region_inscribed_radii(labels, regions)
        protected = protected_feature_region_ids(
            regions, labels, palette, eye_mask, mouth_mask, eye_disks_work,
        )
        ids = list(range(1, len(colors)))
        areas = {rid: int((regions == rid).sum()) for rid in ids}
        if eye_mask is not None:
            eye_frac = ndimage.mean(eye_mask, regions, ids)
        else:
            eye_frac = np.zeros(len(ids))
        if mouth_mask is not None:
            mouth_frac = ndimage.mean(mouth_mask, regions, ids)
        else:
            mouth_frac = np.zeros(len(ids))
        tiny = []
        for rid in ids:
            if rid in protected:
                continue
            if not _region_tiny_for_merge(
                radii.get(rid, 0), colors[rid], px_to_pt, style, paintable_r,
            ):
                continue
            ef, mf = float(eye_frac[rid - 1]), float(mouth_frac[rid - 1])
            if (ef > 0.12 or mf > 0.12) and radii.get(rid, 0) >= paintable_r * 0.22:
                continue
            tiny.append(rid)
        if not tiny:
            break
        tiny.sort(key=lambda rid: areas[rid])
        rid = tiny[0]
        sl = ndimage.find_objects(regions)[rid - 1]
        if sl is None:
            break
        sl = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)
        mask = regions[sl] == rid
        sub_labels = labels[sl]
        sub_regions = regions[sl]
        ring = sub_labels[ndimage.binary_dilation(mask) & ~mask]
        ring = ring[ring != colors[rid]]
        if len(ring) == 0:
            break
        neighbor_colors = np.unique(ring)
        ci = colors[rid]
        best = min(neighbor_colors, key=lambda c: _lab_dist(palette, ci, int(c)))
        sub_labels[mask] = best
        labels[sl] = sub_labels
        merged += 1
    regions, colors = label_regions(labels)
    post_count = int(regions.max())
    colors_used = len(np.unique(labels))
    return labels, regions, colors, {
        "regions_before_merge": pre_count,
        "regions_after_merge": post_count,
        "merge_steps": merged,
        "colors_used": colors_used,
    }


def _region_anchor(labels, regions, rid):
    m = regions == rid
    sub = np.where(m, interior_distance(labels), 0)
    cy, cx = np.unravel_index(int(sub.argmax()), sub.shape)
    return float(cx + 0.5), float(cy + 0.5)


def _search_callout(
    dot_x,
    dot_y,
    text,
    size,
    style,
    px_to_pt,
    placed_boxes,
    cell_labels,
    cell_w,
    cell_h,
    max_leader,
    region_areas,
):
    best = None
    best_score = 1e18
    size_use = max(MIN_LEGIBLE_PT, min(size, style.max_pt))
    for ang in range(0, 360, 12):
        rad = math.radians(ang)
        ux, uy = math.cos(rad), math.sin(rad)
        for step in np.linspace(4.0, max_leader, num=18):
            nx = dot_x + ux * step
            ny = dot_y + uy * step
            if nx < 1 or ny < 1 or nx >= cell_w - 1 or ny >= cell_h - 1:
                continue
            box = _number_aabb_work(nx, ny, text, size_use, style.font, px_to_pt)
            if any(_aabb_overlap(box, other) for other in placed_boxes):
                continue
            if any(_segment_intersects_aabb(dot_x, dot_y, nx, ny, other) for other in placed_boxes):
                continue
            ci = int(cell_labels[int(ny), int(nx)])
            host = region_areas.get(ci, 0)
            score = step - 0.002 * host + 0.5 * abs(ang % 90 - 45)
            if score < best_score:
                best_score = score
                best = (nx, ny, size_use)
    return best


def place_cell_labels(
    cell_labels,
    cell_regions,
    px_to_pt,
    style,
    ppi,
    scale,
    priority_masks=(),
):
    """Inside numbers, then callouts; return placements and guide region ids."""
    dist = interior_distance(cell_labels)
    paintable_r = paintable_radius_work_px(ppi, scale)
    max_leader = max_leader_work_px(ppi, scale)
    h, w = cell_labels.shape
    region_areas = {}
    for rid in np.unique(cell_regions):
        if rid <= 0:
            continue
        ci = int(cell_labels[cell_regions == rid][0])
        region_areas[rid] = region_areas.get(rid, 0) + int((cell_regions == rid).sum())
    color_area = {}
    for ci in np.unique(cell_labels):
        color_area[int(ci)] = int((cell_labels == ci).sum())

    placements: list[NumberPlacement] = []
    placed_boxes = []
    labeled: set[int] = set()

    for k in np.unique(cell_labels):
        comp, count = ndimage.label(cell_labels == k)
        if not count:
            continue
        idx = np.arange(1, count + 1)
        radii = ndimage.maximum(dist, comp, idx)
        spots = ndimage.maximum_position(dist, comp, idx)
        text = str(k + 1)
        for radius, (y, x) in zip(radii, spots):
            rid = int(cell_regions[y, x])
            size = legible_number_pt(float(radius), text, px_to_pt, style)
            if size is None:
                continue
            xw, yw = x + 0.5, y + 0.5
            box = _number_aabb_work(xw, yw, text, size, style.font, px_to_pt)
            if any(_aabb_overlap(box, other) for other in placed_boxes):
                continue
            placements.append(NumberPlacement(rid, text, size, xw, yw))
            placed_boxes.append(box)
            labeled.add(rid)

    covered = np.zeros(cell_labels.shape, dtype=bool)
    for m in priority_masks:
        if m is not None:
            covered |= m >= 0.35

    pending = []
    radii_map = region_inscribed_radii(cell_labels, cell_regions)
    for rid in np.unique(cell_regions):
        rid = int(rid)
        if rid <= 0 or rid in labeled:
            continue
        ci = int(cell_labels[cell_regions == rid][0])
        text = str(ci + 1)
        radius = radii_map.get(rid, 0)
        if radius < paintable_r * 0.35:
            continue
        dot_x, dot_y = _region_anchor(cell_labels, cell_regions, rid)
        pending.append((rid, text, dot_x, dot_y, radius))

    pending.sort(key=lambda t: t[4])
    for rid, text, dot_x, dot_y, radius in pending:
        size = legible_number_pt(radius, text, px_to_pt, style) or MIN_LEGIBLE_PT
        hit = _search_callout(
            dot_x, dot_y, text, size, style, px_to_pt, placed_boxes,
            cell_labels, w, h, max_leader, color_area,
        )
        if hit is None:
            continue
        nx, ny, size_use = hit
        placements.append(NumberPlacement(rid, text, size_use, nx, ny, dot_x, dot_y))
        placed_boxes.append(_number_aabb_work(nx, ny, text, size_use, style.font, px_to_pt))
        labeled.add(rid)

    guide = {int(r) for r in np.unique(cell_regions) if r > 0} - labeled
    return placements, labeled, guide


def collect_all_placements(labels, regions, layout, job, style, face_mask=None, eye_mask=None, mouth_mask=None):
    k = job["scale"] / job["ppi"]
    px_to_pt = k * PT
    ppi = job["ppi"]
    scale = job["scale"]
    all_place: list[NumberPlacement] = []
    labeled_all: set[int] = set()
    guide_all: set[int] = set()
    callout_all: set[int] = set()
    h, w = labels.shape
    for r in range(layout.rows):
        for c in range(layout.cols):
            sx0, sy0 = c * layout.step_x, r * layout.step_y
            wx0, wy0 = sx0 / k, sy0 / k
            x0p, y0p = max(0, int(math.floor(wx0))), max(0, int(math.floor(wy0)))
            cell_x1 = w if c == layout.cols - 1 else min(int(math.ceil((sx0 + layout.step_x) / k)), w)
            cell_y1 = h if r == layout.rows - 1 else min(int(math.ceil((sy0 + layout.step_y) / k)), h)
            cell_labels = labels[y0p:cell_y1, x0p:cell_x1]
            cell_regions = regions[y0p:cell_y1, x0p:cell_x1]
            masks = []
            if face_mask is not None:
                masks.append(face_mask[y0p:cell_y1, x0p:cell_x1])
            if eye_mask is not None:
                masks.append(eye_mask[y0p:cell_y1, x0p:cell_x1])
            if mouth_mask is not None:
                masks.append(mouth_mask[y0p:cell_y1, x0p:cell_x1])
            place, labeled, guide = place_cell_labels(
                cell_labels, cell_regions, px_to_pt, style, ppi, scale, masks,
            )
            for off_y, off_x in ((y0p, x0p),):
                for p in place:
                    if p.is_callout:
                        callout_all.add(p.region_id)
                        all_place.append(
                            NumberPlacement(
                                p.region_id, p.text, p.size,
                                p.num_x + off_x, p.num_y + off_y,
                                p.dot_x + off_x, p.dot_y + off_y,
                            )
                        )
                    else:
                        all_place.append(
                            NumberPlacement(
                                p.region_id, p.text, p.size,
                                p.num_x + off_x, p.num_y + off_y,
                            )
                        )
            labeled_all |= labeled
            guide_all |= guide
    inside = {p.region_id for p in all_place if not p.is_callout}
    return all_place, labeled_all, guide_all, inside, callout_all