#!/usr/bin/env python3
"""Turn a picture into a paint-by-number mural kit for a group.

When a person or animal face is found, face priority is on by default: the face
keeps finer regions and a reserved share of the colors; the background stays
simpler. Inside the face, an eye tier (from YuNet landmarks) keeps still-finer
detail and 2-3 reserved colors so pupils, iris, and glasses frames survive.
Use --no-face-priority for uniform detail, or --face-box x0,y0,x1,y1
(source pixels, after EXIF transpose) to set the face by hand.

Writes to --out:
  mural.pdf         cover + placement map + numbered letter sheets (tracing templates at --scale)
  paint_areas.json  per-color hex, share and wall square feet (input for paint_math.py)
  preview.png       the color-reduced image at wall proportions

With --preview-only also writes:
  preview_kit.pdf      cover + placement map + sample letter sheets (same drawing as the full build)
  preview_outlined.png whole mural with region outlines, numbers, and a faint sheet grid
  preview_pages/       rasterized pages from preview_kit.pdf (when pdftoppm is available)
"""
import argparse
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas
from scipy import ndimage
from sklearn.cluster import KMeans

PT = 72.0                 # PDF points per inch
MARGIN = 0.25             # printer-safe page margin, inches
HEADER, FOOTER = 0.25, 0.45
TARGET_PPI = 24           # work pixels per wall inch
MAX_WORK_PIXELS = 12_000_000
SEED = 0

BAND = HexColor("#e3ecfb")      # overlap band tint
OFF_WALL = HexColor("#eeeeee")  # paper area beyond the wall edge
MARK = HexColor("#d0108a")      # registration crosshairs
INFO = HexColor("#1f4fbf")


@dataclass(frozen=True)
class Style:
    line_pt: float
    line_gray: float
    font: str
    min_pt: float
    max_pt: float


STYLES = {
    # Thin lines and small numbers trace cleanly through graphite or pencil-on-back.
    "trace": Style(line_pt=0.6, line_gray=0.15, font="Helvetica", min_pt=7, max_pt=14),
    # Bold lines are easy to cut along or paint up to.
    "template": Style(line_pt=2.0, line_gray=0.0, font="Helvetica-Bold", min_pt=9, max_pt=24),
}

# Skill detail levels: wall murals use fixed min-region inches; small canvases use % of short side.
DETAIL_LEVELS = {
    "low": {"colors": 8, "wall_min_region_in": 3.0, "canvas_min_region_pct": 2.8, "prep": "soften"},
    "medium": {"colors": 12, "wall_min_region_in": 2.0, "canvas_min_region_pct": 1.9, "prep": "balance"},
    "high": {"colors": 20, "wall_min_region_in": 1.2, "canvas_min_region_pct": 1.1, "prep": "sharpen"},
    "photo": {"colors": 30, "wall_min_region_in": 0.8, "canvas_min_region_pct": 0.8, "prep": "photo"},
}
CANVAS_SHORT_FT = 4.0
PREVIEW_JOB_VERSION = 1
TRACE_SIGMA_FRAC = 0.10   # cap smoothing vs smallest paintable feature
PREVIEW_JOB_NAME = "preview_job.json"
LABELS_ARCHIVE = "labels.npz"


@dataclass(frozen=True)
class Layout:
    landscape: bool
    page_w: float   # inches
    page_h: float
    cw: float       # tile (content) size on paper, inches
    ch: float
    overlap: float
    cols: int
    rows: int

    @property
    def step_x(self):
        return self.cw - self.overlap

    @property
    def step_y(self):
        return self.ch - self.overlap

    @property
    def count(self):
        return self.cols * self.rows


def is_canvas_job(wall_w_ft, wall_h_ft):
    return min(wall_w_ft, wall_h_ft) < CANVAS_SHORT_FT


def lab_merge_min_dist(detail, n_colors, wall_w_ft=None, wall_h_ft=None):
    """Allow closer Lab centers at higher detail so photo never collapses below high."""
    if detail:
        if detail == "photo" and wall_w_ft is not None and not is_canvas_job(wall_w_ft, wall_h_ft):
            return 1.2
        return {"low": 3.5, "medium": 3.0, "high": 2.2, "photo": 1.0}[detail]
    return max(1.5, 4.0 - 0.085 * n_colors)


def min_region_for_detail(detail, wall_w_ft, wall_h_ft):
    spec = DETAIL_LEVELS[detail]
    if is_canvas_job(wall_w_ft, wall_h_ft):
        short_in = min(wall_w_ft, wall_h_ft) * 12
        return short_in * spec["canvas_min_region_pct"] / 100.0
    return spec["wall_min_region_in"]


def prep_for_detail(detail):
    return DETAIL_LEVELS[detail]["prep"] if detail else "none"


def prep_choices():
    return ("auto", "none", "soften", "balance", "sharpen", "photo")


def apply_image_prep(img, prep):
    if prep == "none":
        return img
    if prep == "soften":
        base = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        return ImageOps.autocontrast(base, cutoff=1)
    if prep == "balance":
        return ImageOps.autocontrast(img, cutoff=1)
    if prep == "sharpen":
        return ImageOps.autocontrast(img.filter(ImageFilter.SHARPEN), cutoff=0.5)
    if prep == "photo":
        base = ImageOps.autocontrast(img, cutoff=0.5)
        return base.filter(ImageFilter.UnsharpMask(radius=1.2, percent=140, threshold=3))
    raise ValueError(f"unknown prep {prep!r}")


def apply_detail_and_prep_defaults(a):
    if a.detail:
        spec = DETAIL_LEVELS[a.detail]
        if a.colors is None:
            a.colors = spec["colors"]
        if a.min_region_in is None:
            a.min_region_in = min_region_for_detail(a.detail, a.wall_width_ft, a.wall_height_ft)
    if a.prep == "auto":
        a.prep = prep_for_detail(a.detail) if a.detail else "none"


def merge_from_preview_args(a, parser):
    job_path = a.from_preview / PREVIEW_JOB_NAME
    if not job_path.is_file():
        parser.error(f"--from-preview missing {PREVIEW_JOB_NAME}")
    saved = json.loads(job_path.read_text())
    if saved.get("version") != PREVIEW_JOB_VERSION:
        parser.error(f"unsupported {PREVIEW_JOB_NAME} version")
    a._from_preview_dir = a.from_preview
    cli = saved.get("argv", {})
    path_keys = {"image", "palette_json"}
    for key, val in cli.items():
        if key in ("out", "preview_only", "preview_sheets", "from_preview"):
            continue
        if getattr(a, key, None) is None and val is not None:
            if key in path_keys:
                setattr(a, key, Path(val))
            else:
                setattr(a, key, val)
    if a.wall_width_ft is None:
        a.wall_width_ft = saved["wall_width_ft"]
    if a.wall_height_ft is None:
        a.wall_height_ft = saved["wall_height_ft"]
    if a.colors is None:
        a.colors = saved["colors"]
    if a.min_region_in is None:
        a.min_region_in = saved["min_region_in"]


def trace_sigma(ppi, min_region_in, face=None):
    raw = max(1.0, 0.1 * ppi)
    finest_in = float(min_region_in)
    if face and face.get("eye_min_region_in"):
        finest_in = min(finest_in, float(face["eye_min_region_in"]))
    elif face and face.get("face_min_region_in"):
        finest_in = min(finest_in, float(face["face_min_region_in"]))
    cap = max(1.0, TRACE_SIGMA_FRAC * finest_in * ppi)
    return min(raw, cap)


def label_metrics(labels, regions, face_mask, eye_mask):
    n_reg = int(regions.max())
    face_pix = float((face_mask >= 0.35).sum()) if face_mask is not None else 0.0
    eye_pix = float((eye_mask >= 0.35).sum()) if eye_mask is not None else 0.0
    face_regs, eye_regs = set(), set()
    if face_pix:
        face_regs = set(np.unique(regions[face_mask >= 0.35]).tolist()) - {0}
    if eye_pix:
        eye_regs = set(np.unique(regions[eye_mask >= 0.35]).tolist()) - {0}
    return {
        "regions": n_reg,
        "face_regions": len(face_regs),
        "eye_regions": len(eye_regs),
        "face_regions_per_10k_px": round(len(face_regs) / face_pix * 10_000, 2) if face_pix else 0.0,
    }


def save_preview_job(out, a, resolved, face, labels, palette, regions, crop_box, face_mask, eye_mask):
    argv = {
        "image": str(a.image) if a.image else None,
        "wall_width_ft": a.wall_width_ft,
        "wall_height_ft": a.wall_height_ft,
        "colors": a.colors,
        "coats": a.coats,
        "coverage_sqft_per_gal": a.coverage_sqft_per_gal,
        "overlap_in": a.overlap_in,
        "min_region_in": a.min_region_in,
        "transfer": a.transfer,
        "brand": a.brand,
        "scale": a.scale,
        "no_face_priority": a.no_face_priority,
        "face_box": a.face_box,
        "detail": a.detail,
        "prep": resolved.get("prep"),
        "sheet_tint": a.sheet_tint,
    }
    meta = {
        "version": PREVIEW_JOB_VERSION,
        "argv": argv,
        "wall_width_ft": a.wall_width_ft,
        "wall_height_ft": a.wall_height_ft,
        "colors": int(len(palette)),
        "min_region_in": a.min_region_in,
        "detail": a.detail,
        "prep": resolved.get("prep"),
        "crop_box": list(crop_box),
        "ppi": resolved["ppi"],
        "fit_note": resolved["fit_note"],
        "face_priority": {k: v for k, v in face.items() if k not in ("mask", "eye_mask")},
        "label_metrics": label_metrics(labels, regions, face_mask, eye_mask),
    }
    (out / PREVIEW_JOB_NAME).write_text(json.dumps(meta, indent=2) + "\n")
    npz = {k: arr for k, arr in (
        ("labels", labels),
        ("regions", regions),
        ("palette", palette),
    ) if arr is not None}
    if face_mask is not None:
        npz["face_mask"] = face_mask.astype(np.float32)
    if eye_mask is not None:
        npz["eye_mask"] = eye_mask.astype(np.float32)
    np.savez_compressed(out / LABELS_ARCHIVE, **npz)


def load_saved_labels(from_dir):
    path = from_dir / LABELS_ARCHIVE
    if not path.is_file():
        raise SystemExit(f"missing {LABELS_ARCHIVE} in {from_dir}")
    data = np.load(path)
    labels = data["labels"]
    palette = data["palette"]
    regions = data["regions"]
    face_mask = data["face_mask"] if "face_mask" in data else None
    eye_mask = data["eye_mask"] if "eye_mask" in data else None
    return labels, palette, regions, face_mask, eye_mask


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, help="source picture (required unless --from-preview)")
    p.add_argument("--from-preview", type=Path,
                   help="build the full kit from a prior --preview-only directory (preview_job.json + labels.npz)")
    p.add_argument("--wall-width-ft", type=float, help="wall size (from preview job when --from-preview)")
    p.add_argument("--wall-height-ft", type=float)
    p.add_argument("--colors", type=int, help="paint colors; set automatically with --detail")
    p.add_argument("--detail", choices=sorted(DETAIL_LEVELS),
                   help="skill detail level (sets --colors and --min-region-in when not passed)")
    p.add_argument("--prep", choices=prep_choices(), default="auto",
                   help="source image prep before tiling (auto follows --detail)")
    p.add_argument("--coats", type=int, choices=(1, 2), default=2, help="recorded for paint_math.py and shown on the cover")
    p.add_argument("--coverage-sqft-per-gal", type=float, default=375.0, help="recorded for paint_math.py")
    p.add_argument("--overlap-in", type=float, default=0.5, help="overlap between neighboring sheets, paper inches")
    p.add_argument("--min-region-in", type=float,
                   help="wall inches; regions smaller than this squared, or thinner than half of it, are merged away")
    p.add_argument("--sheet-tint", type=float, default=0.0,
                   help="0..0.35 light palette fill under sheet outlines so sheets match preview")
    p.add_argument("--transfer", choices=sorted(STYLES), default="trace")
    p.add_argument("--brand", default="", help="paint brand name, used for labels only")
    p.add_argument("--palette-json", type=Path, help="brand matches keyed by color number (see brand_match.py)")
    p.add_argument("--scale", type=float, default=1.0, help="print scale; 1.0 = sheets are true wall size")
    p.add_argument("--preview-only", action="store_true",
                   help="write cover, map, areas and a rich preview (no full mural.pdf sheets)")
    p.add_argument("--preview-sheets", type=int, default=2,
                   help="with --preview-only, sample sheets in preview_kit.pdf (0 = map and outlined image only)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--no-face-priority", action="store_true",
                   help="disable automatic face priority (uniform min-region and palette)")
    p.add_argument("--face-box", default="",
                   help="manual face box in source pixels (after EXIF transpose): x0,y0,x1,y1")
    a = p.parse_args()
    a._from_preview_dir = None
    if a.from_preview:
        merge_from_preview_args(a, p)
    else:
        if not a.image:
            p.error("--image required unless --from-preview")
        if a.wall_width_ft is None or a.wall_height_ft is None:
            p.error("--wall-width-ft and --wall-height-ft are required")
        apply_detail_and_prep_defaults(a)
    if a.colors is None or a.colors < 2:
        p.error("need --colors or --detail (at least 2 colors)")
    if a.min_region_in is None:
        a.min_region_in = 3.0
    if min(a.wall_width_ft, a.wall_height_ft, a.scale, a.min_region_in, a.coverage_sqft_per_gal) <= 0:
        p.error("sizes, --scale, --min-region-in and coverage must be positive")
    if not 0 <= a.sheet_tint <= 0.35:
        p.error("--sheet-tint must be between 0 and 0.35")
    if not 0 <= a.overlap_in < 2:
        p.error("--overlap-in must be between 0 and 2 inches")
    if a.preview_sheets < 0:
        p.error("--preview-sheets must be >= 0")
    if a.face_box:
        try:
            parse_face_box(a.face_box)
        except ValueError as e:
            p.error(f"--face-box: {e}")
    return a


# ---------- face priority ----------

YUNET_MODEL = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
FACE_PAD = 1.6                 # expand detector box to include hairline, ears, beard/chin
FACE_EXTRA_TOP = 0.10          # extra height upward (hair / ears), as a fraction of the padded box
FACE_COLOR_SHARE = 0.40        # share of the color budget reserved from face pixels
FACE_MIN_COLORS = 4            # at least this many reserved tones (Low needs 4)
FACE_REGION_FRAC = 0.30        # face min-region is this fraction of the level's --min-region-in
FACE_REGION_FLOOR_IN = 0.12    # still paintable with a small brush on a canvas
YUNET_SCORE = 0.6
FACE_FEATHER = 0.18            # soft edge as a fraction of box size, so no hard rectangle
EYE_RADIUS_FRAC = 0.55         # eye-box radius as a fraction of inter-eye distance (covers lens + frame)
EYE_FEATHER = 0.22             # soft disk edge, as a fraction of the radius
EYE_COLORS = 3                 # floor reserved from eye pixels once N>=10 (pupil/iris, white/lens, frame); scales up with N, cap 8
EYE_REGION_FRAC = 0.40         # eye min-region is this fraction of the face min-region
EYE_REGION_FLOOR_IN = 0.08     # still label-able with a small number on a canvas


def parse_face_box(text):
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("need x0,y0,x1,y1")
    x0, y0, x1, y1 = (float(p) for p in parts)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("x1,y1 must be greater than x0,y0")
    return (x0, y0, x1, y1)


def _pad_box(box, img_w, img_h, scale=FACE_PAD, extra_top=FACE_EXTRA_TOP):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = (x1 - x0) * scale, (y1 - y0) * scale
    y0p = cy - bh / 2 - extra_top * bh
    y1p = cy + bh / 2
    x0p, x1p = cx - bw / 2, cx + bw / 2
    return (max(0, x0p), max(0, y0p), min(img_w, x1p), min(img_h, y1p))


def _detect_faces_yunet(rgb_img):
    """Return a list of dicts {box, score, landmarks} in source pixels. Landmarks are
    (right eye, left eye, nose, right mouth, left mouth). Empty if the model is missing or none are found."""
    if not YUNET_MODEL.is_file():
        print(f"face priority: YuNet model not found at {YUNET_MODEL}; no automatic detection")
        return []
    try:
        import os
        os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
        import cv2
    except ImportError:
        print("face priority: OpenCV is not installed; no automatic detection")
        return []
    arr = np.asarray(rgb_img)
    h, w = arr.shape[:2]
    # Detect at a moderate size so small faces in large photos still fall in YuNet's range.
    long = max(w, h)
    scale = 1600 / long if long > 1600 else 1.0
    dw, dh = int(round(w * scale)), int(round(h * scale))
    bgr = arr[:, :, ::-1].copy()
    if scale != 1.0:
        bgr = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA)
    det = cv2.FaceDetectorYN_create(str(YUNET_MODEL), "", (dw, dh), YUNET_SCORE, 0.3, 5000)
    _retval, faces = det.detect(bgr)
    if faces is None or len(faces) == 0:
        return []
    out = []
    inv = 1.0 / scale
    for f in faces:
        x, y, fw, fh, score = float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14])
        if score < YUNET_SCORE or fw < 8 or fh < 8:
            continue
        lms = [(float(f[4 + j * 2]) * inv, float(f[5 + j * 2]) * inv) for j in range(5)]
        out.append({
            "box": (x * inv, y * inv, (x + fw) * inv, (y + fh) * inv),
            "score": score,
            "landmarks": lms,
        })
    return out


def _map_box_to_work(src_box, crop_box, work_size):
    x0, y0, x1, y1 = src_box
    cx0, cy0, cx1, cy1 = crop_box
    # Intersect with the wall crop, then scale into work pixels.
    ix0, iy0 = max(x0, cx0), max(y0, cy0)
    ix1, iy1 = min(x1, cx1), min(y1, cy1)
    if ix1 - ix0 < 2 or iy1 - iy0 < 2:
        return None
    cw, ch = cx1 - cx0, cy1 - cy0
    ww, wh = work_size
    return (
        (ix0 - cx0) / cw * ww,
        (iy0 - cy0) / ch * wh,
        (ix1 - cx0) / cw * ww,
        (iy1 - cy0) / ch * wh,
    )


def _map_point_to_work(x, y, crop_box, work_size):
    cx0, cy0, cx1, cy1 = crop_box
    cw, ch = cx1 - cx0, cy1 - cy0
    ww, wh = work_size
    return ((x - cx0) / cw * ww, (y - cy0) / ch * wh)


def _eye_disks_from_landmarks(landmarks, img_w, img_h):
    """Return a list of (cx, cy, radius) in the given pixel space, one disk per eye."""
    if not landmarks or len(landmarks) < 2:
        return []
    e0, e1 = landmarks[0], landmarks[1]
    dist = math.hypot(e1[0] - e0[0], e1[1] - e0[1])
    if dist < 8:
        return []
    r = EYE_RADIUS_FRAC * dist
    out = []
    for ex, ey in (e0, e1):
        if 0 <= ex < img_w and 0 <= ey < img_h:
            out.append((ex, ey, r))
    return out


def _estimate_eye_disks(face_box):
    """Fallback when landmarks are missing (manual --face-box): two disks in the upper face."""
    x0, y0, x1, y1 = face_box
    bw, bh = x1 - x0, y1 - y0
    r = 0.16 * bw
    cy = y0 + 0.38 * bh
    return [(x0 + 0.32 * bw, cy, r), (x0 + 0.68 * bw, cy, r)]


def _soft_disks(h, w, disks, feather=EYE_FEATHER):
    """Float mask 0..1 for circular eye regions, cosine-feathered at the rim."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    for cx, cy, r in disks:
        r = max(r, 1.0)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        inner = r * (1.0 - feather)
        ramp = np.where(
            d <= inner, 1.0,
            np.where(d >= r, 0.0, 0.5 - 0.5 * np.cos(np.clip((r - d) / (r - inner), 0, 1) * np.pi)),
        )
        mask = np.maximum(mask, ramp.astype(np.float32))
    return np.clip(mask, 0, 1)


def _maybe_upscale_ppi(src, wall_w_in, wall_h_in, ppi, enabled):
    """If the source is low-res for this wall, raise work PPI (Lanczos upscale in fit_to_wall)."""
    if not enabled:
        return ppi
    w, h = src.size
    target = wall_w_in / wall_h_in
    if math.isclose(w / h, target, rel_tol=0.005):
        cw, ch = w, h
    elif w / h > target:
        cw, ch = round(h * target), h
    else:
        cw, ch = w, round(w / target)
    src_ppi = min(cw / wall_w_in, ch / wall_h_in)
    max_ppi = math.sqrt(MAX_WORK_PIXELS / (wall_w_in * wall_h_in))
    bumped = min(TARGET_PPI * 2, src_ppi * 2, max_ppi)
    if bumped > ppi * 1.15:
        print(f"face priority: work resolution {bumped:.1f} px/in "
              f"(Lanczos upscale; source is {src_ppi:.1f} px/in)")
        return bumped
    return ppi


def _soft_mask(h, w, boxes, feather=FACE_FEATHER):
    """Float mask 0..1 with a cosine ramp at each box edge so the face rectangle doesn't print through."""
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.zeros((h, w), dtype=np.float32)
    for x0, y0, x1, y1 in boxes:
        bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        fx, fy = feather * bw, feather * bh
        # Distance "inside" each edge, in pixels; 0 at the edge, positive inward.
        left = xx - x0
        right = x1 - xx
        top = yy - y0
        bottom = y1 - yy
        inside = (left >= 0) & (right >= 0) & (top >= 0) & (bottom >= 0)
        if not np.any(inside):
            continue
        edge = np.minimum(np.minimum(left, right) / fx, np.minimum(top, bottom) / fy)
        ramp = 0.5 - 0.5 * np.cos(np.clip(edge, 0, 1) * np.pi)
        mask = np.maximum(mask, np.where(inside, ramp, 0).astype(np.float32))
    return np.clip(mask, 0, 1)


def _eye_color_count(n, n_face):
    """Reserve 2-3 eye colors at Low, more at High/Photo, always within the same total N."""
    if n < 6:
        return 0
    if n < 10:
        want = 2
    else:
        # Low 10 -> 3, Medium 20 -> 4, High 35 -> 7, Photo 60 -> 8 (cap)
        want = min(8, max(EYE_COLORS, int(round(n * 0.20))))
    return min(want, max(0, n - n_face - 2))


def resolve_faces(src, work_img, crop_box, args):

    """Detect or apply a manual face box. Returns a dict; mask is None when face priority is off or no face is found."""
    info = {"enabled": not args.no_face_priority, "found": False, "detector": "none",
            "faces": [], "mask": None, "face_min_region_in": None, "face_colors_reserved": 0,
            "eye_mask": None, "eye_min_region_in": None, "eye_colors_reserved": 0,
            "eye_count": 0, "eye_disks_work": []}
    if args.no_face_priority:
        print("face priority: off")
        return info
    src_w, src_h = src.size
    work_w, work_h = work_img.size
    detections, detector = [], "none"
    if args.face_box:
        detections = [{"box": parse_face_box(args.face_box), "score": 1.0, "landmarks": []}]
        detector = "manual"
    else:
        detections = _detect_faces_yunet(src)
        if detections:
            detector = "yunet"
    if not detections:
        print("face priority: no face found; using uniform detail")
        return info
    # Keep the strongest face(s). Drop a weaker detection whose center sits inside
    # a higher-scoring padded box (a cat's chest, a highlight) so group photos still
    # keep two clearly separate faces.
    ranked = sorted(detections, key=lambda d: -d["score"])
    kept, kept_padded = [], []
    for det in ranked:
        box = det["box"]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if any(px0 <= cx <= px1 and py0 <= cy <= py1 for px0, py0, px1, py1 in kept_padded):
            continue
        padded = _pad_box(box, src_w, src_h)
        kept.append(det)
        kept_padded.append(padded)
    work_boxes, faces_meta, eye_disks_work = [], [], []
    for det, padded in zip(kept, kept_padded):
        mapped = _map_box_to_work(padded, crop_box, (work_w, work_h))
        lms = det.get("landmarks") or []
        lms_work = [_map_point_to_work(x, y, crop_box, (work_w, work_h)) for x, y in lms] if lms else []
        disks_src = _eye_disks_from_landmarks(lms, src_w, src_h) if lms else []
        if not disks_src:
            disks_src = _estimate_eye_disks(padded)
        for cx, cy, r in disks_src:
            wcx, wcy = _map_point_to_work(cx, cy, crop_box, (work_w, work_h))
            wr = r * (work_w / (crop_box[2] - crop_box[0]))
            if wr >= 2:
                eye_disks_work.append((wcx, wcy, wr))
        faces_meta.append({
            "source_box": [round(v, 1) for v in det["box"]],
            "padded_source_box": [round(v, 1) for v in padded],
            "work_box": None if mapped is None else [round(v, 1) for v in mapped],
            "landmarks_source": [[round(x, 1), round(y, 1)] for x, y in lms],
            "landmarks_work": [[round(x, 1), round(y, 1)] for x, y in lms_work],
            "score": round(float(det["score"]), 3),
        })
        if mapped is not None:
            work_boxes.append(mapped)
    if not work_boxes:
        print("face priority: a face was found but it sits outside the wall crop; using uniform detail")
        info["detector"] = detector
        info["faces"] = faces_meta
        return info
    face_min = max(FACE_REGION_FLOOR_IN, FACE_REGION_FRAC * args.min_region_in)
    n_face = min(args.colors - 2, max(FACE_MIN_COLORS, int(round(FACE_COLOR_SHARE * args.colors))))
    n_face = max(1, n_face)
    n_eye = _eye_color_count(args.colors, n_face)
    eye_min = max(EYE_REGION_FLOOR_IN, EYE_REGION_FRAC * face_min) if n_eye else None
    eye_mask = _soft_disks(work_h, work_w, eye_disks_work) if eye_disks_work and n_eye else None
    info.update({
        "found": True, "detector": detector, "faces": faces_meta,
        "mask": _soft_mask(work_h, work_w, work_boxes),
        "face_min_region_in": round(face_min, 3), "face_colors_reserved": n_face,
        "eye_mask": eye_mask,
        "eye_min_region_in": None if eye_min is None else round(eye_min, 3),
        "eye_colors_reserved": n_eye,
        "eye_count": 0 if eye_mask is None else len(eye_disks_work),
        "eye_disks_work": [[round(cx, 1), round(cy, 1), round(r, 1)] for cx, cy, r in eye_disks_work],
    })
    bits = ", ".join(
        f"score {f['score']}" + (f", work box {f['work_box']}" if f["work_box"] else ", outside crop")
        for f in faces_meta
    )
    msg = (f"face priority: {len(work_boxes)} face(s) ({detector}, {bits}); "
           f"reserved {n_face} of {args.colors} colors, face min-region {face_min:g} in")
    if eye_mask is not None:
        msg += (f"; eye priority: {len(eye_disks_work)} eye(s), reserved {n_eye} colors, "
                f"eye min-region {eye_min:g} in")
    print(msg)
    return info


def _kmeans_lab(pixels, n, seed=SEED, weight_exp=0.5):
    """Histogram-weighted k-means in Lab. pixels is Nx3 uint8. Returns cluster centers in Lab."""
    if n < 1 or len(pixels) == 0:
        return np.zeros((0, 3))
    bins = (pixels >> 3).astype(np.int32) @ np.array([1024, 32, 1])
    uniq, inverse, counts = np.unique(bins, return_inverse=True, return_counts=True)
    n = int(min(n, len(uniq)))
    if n < 1:
        return np.zeros((0, 3))
    means = np.zeros((len(uniq), 3))
    np.add.at(means, inverse, pixels)
    lab = rgb_to_lab(means / counts[:, None])
    w = np.power(counts.astype(np.float64), weight_exp)
    if n == 1:
        # Weighted mean of the bins.
        return (lab * w[:, None]).sum(axis=0, keepdims=True) / w.sum()
    km = KMeans(n_clusters=n, n_init=4, random_state=seed).fit(lab, sample_weight=w)
    return km.cluster_centers_


def _merge_close_lab(centers, min_dist=3.5, target_count=0):
    kept = []
    for c in centers:
        if all(np.linalg.norm(c - k) >= min_dist for k in kept):
            kept.append(c)
    out = np.array(kept) if kept else np.zeros((0, 3))
    if target_count > len(out) and len(out) < len(centers):
        extra = [c for c in centers
                 if not any(np.linalg.norm(c - k) < min_dist for k in out)]
        for c in extra:
            if len(out) >= target_count:
                break
            if all(np.linalg.norm(c - k) >= min_dist * 0.55 for k in out):
                out = np.vstack([out, c])
    return out


def _assign_nearest_lab(rgb, centers_lab):
    lab = rgb_to_lab(rgb)
    delta = lab[:, :, None, :] - centers_lab[None, None, :, :]
    return np.sqrt((delta * delta).sum(-1)).argmin(-1).astype(np.int32)


def _median_composite(img, face_mask, ppi, eye_mask=None, colors=8):
    """Stronger noise filter in the background, weaker in the face, none in the eyes."""
    bg_k = 2 * int(0.1 * ppi) + 1
    if colors >= 24:
        bg_k = max(3, bg_k - 2)
    face_k = 2 * int(0.04 * ppi) + 1
    orig = np.asarray(img)
    bg = np.asarray(img.filter(ImageFilter.MedianFilter(bg_k)))
    face = orig if face_k < 3 else np.asarray(img.filter(ImageFilter.MedianFilter(max(3, face_k))))
    m_face = face_mask[..., None]
    mixed = face * m_face + bg * (1 - m_face)
    if eye_mask is not None:
        m_eye = eye_mask[..., None]
        mixed = orig * m_eye + mixed * (1 - m_eye)
    return np.rint(mixed).astype(np.uint8)


def _clahe_eyes(rgb, eye_mask, clip=2.0, tile=4):
    """Local contrast lift only inside the eye disks, to pull iris/pupil out of lens glare."""
    try:
        import os
        os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
        import cv2
    except ImportError:
        return rgb
    if eye_mask is None or float(eye_mask.max()) < 0.5:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    lifted = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge([lifted, a, b]), cv2.COLOR_LAB2RGB)
    m = eye_mask[..., None]
    return np.rint(enhanced * m + rgb * (1 - m)).astype(np.uint8)


def quantize_face_priority(img, n, ppi, face_mask, eye_mask=None, n_eye=0, merge_min_dist=3.5):
    """Reserve palette slots for eyes, then the face, then the background; total stays at N.

    Cluster centers are taken from the un-enhanced image so paint colors stay true to the
    photo. Assignment inside the eye disks uses a local CLAHE lift so a dark pupil can
    separate from lens glare, and is locked to the reserved eye centers so it doesn't
    collapse into hair or skin.
    """
    mixed = _median_composite(img, face_mask, ppi, eye_mask, colors=n)
    n_eye = n_eye if eye_mask is not None and n_eye > 0 else 0
    n_face = min(n - n_eye - 2, max(FACE_MIN_COLORS, int(round(FACE_COLOR_SHARE * n))))
    n_face = max(1, min(n_face, n - n_eye - 1))
    n_bg = max(1, n - n_face - n_eye)
    eye_pix = mixed[eye_mask >= 0.5].reshape(-1, 3) if n_eye else np.zeros((0, 3), np.uint8)
    face_pix = mixed[(face_mask >= 0.5) & ((eye_mask < 0.5) if n_eye else True)].reshape(-1, 3)
    bg_pix = mixed[face_mask < 0.5].reshape(-1, 3)
    parts = []
    n_eye_kept = 0
    merge = merge_min_dist
    if n_eye and len(eye_pix) >= n_eye:
        # Lower weight exponent so a small pupil can still earn its own cluster.
        eye_c = _merge_close_lab(_kmeans_lab(eye_pix, n_eye, SEED, weight_exp=0.25), min_dist=merge)
        parts.append(eye_c)
        n_eye_kept = len(eye_c)
    if len(face_pix):
        parts.append(_kmeans_lab(face_pix, n_face, SEED + 1))
    if len(bg_pix):
        parts.append(_kmeans_lab(bg_pix, n_bg, SEED + 2))
    if not parts:
        return quantize(Image.fromarray(mixed), n, ppi, merge_min_dist=merge_min_dist)
    centers = _merge_close_lab(np.vstack(parts), min_dist=merge, target_count=n)
    # Fill slots freed by merging, preferring leftover eye then face variation.
    # New unique eye extras stay at the end of the kept list; count them as eye colors.
    if len(centers) < n and len(eye_pix):
        before = len(centers)
        extra = _kmeans_lab(eye_pix, n_eye + (n - len(centers)), SEED + 3, weight_exp=0.25)
        centers = _merge_close_lab(np.vstack([centers, extra]), min_dist=merge, target_count=n)
        n_eye_kept += max(0, len(centers) - before)
    if len(centers) < n and len(face_pix):
        extra = _kmeans_lab(face_pix, n_face + (n - len(centers)), SEED + 4)
        centers = _merge_close_lab(np.vstack([centers, extra]), min_dist=merge, target_count=n)
    if len(centers) < n:
        extra = _kmeans_lab(mixed.reshape(-1, 3), n, SEED + 5)
        centers = _merge_close_lab(np.vstack([centers, extra]), min_dist=merge, target_count=n)
    if len(centers) > n:
        centers = centers[:n]
        n_eye_kept = min(n_eye_kept, len(centers))
    if len(centers) < 2:
        return quantize(Image.fromarray(mixed), n, ppi, merge_min_dist=merge_min_dist)
    labels = _assign_nearest_lab(mixed, centers)
    # Inside the eye disks, assign against a contrast-lifted view but only to reserved eye colors.
    if n_eye_kept >= 2:
        eye_centers = centers[:n_eye_kept]
        lifted = _clahe_eyes(mixed, eye_mask, clip=2.5, tile=4)
        labels_eye = _assign_nearest_lab(lifted, eye_centers)
        labels = np.where(eye_mask >= 0.5, labels_eye, labels)
    return labels


def majority_filter_face_aware(labels, n, win_bg, win_face, face_mask, eye_mask=None, preserve_bg=False):
    """Fine majority inside the face, none in the eyes, coarser in the background; smooth seams."""
    fine = majority_filter(labels, n, max(1, win_face))
    coarse = majority_filter(labels, n, max(1, win_bg))
    out = np.where(face_mask >= 0.35, fine, labels if preserve_bg else coarse)
    if eye_mask is not None:
        # Leave eye pixels unsmoothed so a pupil or frame line isn't majority-voted away.
        out = np.where(eye_mask >= 0.35, labels, out)
        seam = (face_mask > 0.12) & (face_mask < 0.75) & (eye_mask < 0.35)
    else:
        seam = (face_mask > 0.12) & (face_mask < 0.75)
    if np.any(seam):
        blended = majority_filter(out, n, 3)
        out = np.where(seam, blended, out)
    return out


def clean_regions_face_aware(labels, min_area_px, min_thick_px, face_area_px, face_thick_px,
                             face_mask, eye_mask=None, eye_area_px=None, eye_thick_px=None,
                             max_passes=20):
    """Merge tiny/thin regions, with finer thresholds on the face and finer still in the eyes."""
    if eye_mask is None:
        eye_mask = np.zeros(labels.shape, dtype=np.float32)
        eye_area_px = face_area_px
        eye_thick_px = face_thick_px
    for _ in range(max_passes):
        regions, colors = label_regions(labels)
        ids = np.arange(1, len(colors))
        area = np.bincount(regions.ravel(), minlength=len(colors))[1:]
        thick = 2 * ndimage.maximum(interior_distance(labels), regions, ids)
        face_frac = ndimage.mean(face_mask, regions, ids)
        eye_frac = ndimage.mean(eye_mask, regions, ids)
        # Interpolate through the soft edges so cleanup doesn't form a hard rectangle.
        t_e = np.clip((np.asarray(eye_frac) - 0.1) / 0.5, 0, 1)
        t_f = np.clip((np.asarray(face_frac) - 0.1) / 0.5, 0, 1)
        area_lim = eye_area_px * t_e + face_area_px * t_f * (1 - t_e) + min_area_px * (1 - t_f) * (1 - t_e)
        thick_lim = eye_thick_px * t_e + face_thick_px * t_f * (1 - t_e) + min_thick_px * (1 - t_f) * (1 - t_e)
        bad = ids[(area < area_lim) | (thick < thick_lim)]
        if len(bad) == 0:
            return labels
        slices = ndimage.find_objects(regions)
        changed = False
        for rid in bad[np.argsort(area[bad - 1], kind="stable")]:
            sl = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in slices[rid - 1])
            mask = regions[sl] == rid
            sub = labels[sl]
            ring = sub[ndimage.binary_dilation(mask) & ~mask]
            ring = ring[ring != colors[rid]]
            if len(ring):
                sub[mask] = np.bincount(ring).argmax()
                changed = True
        if not changed:
            return labels
    return labels


# ---------- image -> clean color regions ----------

def fit_to_wall(img, wall_w_in, wall_h_in, ppi):
    """Center-crop to the wall's aspect ratio (no distortion, no padding), then resample to work resolution."""
    w, h = img.size
    target = wall_w_in / wall_h_in
    if math.isclose(w / h, target, rel_tol=0.005):
        box, note = (0, 0, w, h), "used whole (aspect already matches the wall)"
    elif w / h > target:
        new_w = round(h * target)
        box, note = ((w - new_w) // 2, 0, (w - new_w) // 2 + new_w, h), f"center-cropped {100 * (1 - new_w / w):.1f}% of the width"
    else:
        new_h = round(w / target)
        box, note = (0, (h - new_h) // 2, w, (h - new_h) // 2 + new_h), f"center-cropped {100 * (1 - new_h / h):.1f}% of the height"
    size = (round(wall_w_in * ppi), round(wall_h_in * ppi))
    return img.crop(box).resize(size, Image.LANCZOS), note, box


def rgb_to_lab(rgb):
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def quantize(img, n, ppi, merge_min_dist=3.5):
    """K-means in Lab space on a median-filtered copy (removes noise, keeps edges crisp so
    antialiased edge pixels don't become a color of their own).

    Clustering runs on a color histogram weighted by sqrt(pixel count), so small but distinct
    features (a sun, a trunk) keep their own color instead of being outvoted by big gradients.
    """
    filtered = np.asarray(img.filter(ImageFilter.MedianFilter(2 * int(0.1 * ppi) + 1))).reshape(-1, 3)
    bins = (filtered >> 3).astype(np.int32) @ np.array([1024, 32, 1])
    uniq, inverse, counts = np.unique(bins, return_inverse=True, return_counts=True)
    means = np.zeros((len(uniq), 3))
    np.add.at(means, inverse, filtered)
    lab = rgb_to_lab(means / counts[:, None])
    km = KMeans(n_clusters=n, n_init=4, random_state=SEED).fit(lab, sample_weight=np.sqrt(counts))
    return km.labels_[inverse].reshape(img.height, img.width).astype(np.int32)


def majority_filter(labels, n, size):
    best = np.full(labels.shape, -1.0, dtype=np.float32)
    out = labels.copy()
    for k in range(n):
        share = ndimage.uniform_filter((labels == k).astype(np.float32), size=size, mode="nearest")
        win = share > best
        best[win], out[win] = share[win], k
    return out


def interior_distance(labels):
    """Distance (px) from each pixel to the nearest color boundary or array edge."""
    p = np.pad(labels, 1, constant_values=-1)
    c = p[1:-1, 1:-1]
    edge = (c != p[:-2, 1:-1]) | (c != p[2:, 1:-1]) | (c != p[1:-1, :-2]) | (c != p[1:-1, 2:])
    return ndimage.distance_transform_edt(~edge)


def label_regions(labels):
    """Connected same-color regions. Returns (region map with ids 1..R, color of each id; index 0 unused)."""
    regions = np.zeros(labels.shape, dtype=np.int32)
    colors = [-1]
    for k in np.unique(labels):
        lab, count = ndimage.label(labels == k)
        regions[lab > 0] = lab[lab > 0] + len(colors) - 1
        colors += [int(k)] * count
    return regions, np.array(colors)


def clean_regions(labels, min_area_px, min_thick_px, max_passes=20):
    """Merge regions too small or too thin to paint into the neighbor color they touch most."""
    for _ in range(max_passes):
        regions, colors = label_regions(labels)
        ids = np.arange(1, len(colors))
        area = np.bincount(regions.ravel(), minlength=len(colors))[1:]
        thick = 2 * ndimage.maximum(interior_distance(labels), regions, ids)
        bad = ids[(area < min_area_px) | (thick < min_thick_px)]
        if len(bad) == 0:
            return labels
        slices = ndimage.find_objects(regions)
        changed = False
        for rid in bad[np.argsort(area[bad - 1], kind="stable")]:
            sl = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in slices[rid - 1])
            mask = regions[sl] == rid
            sub = labels[sl]
            ring = sub[ndimage.binary_dilation(mask) & ~mask]
            ring = ring[ring != colors[rid]]
            if len(ring):
                sub[mask] = np.bincount(ring).argmax()
                changed = True
        if not changed:
            return labels
    return labels


def finalize_palette(labels, rgb):
    """Renumber used colors 0..m-1 from lightest to darkest; return labels and mean RGB per color."""
    used = np.unique(labels)
    means = np.array([rgb[labels == k].mean(axis=0) for k in used])
    order = np.argsort(-rgb_to_lab(means)[:, 0], kind="stable")
    remap = np.zeros(labels.max() + 1, dtype=np.int32)
    remap[used[order]] = np.arange(len(used))
    return remap[labels], np.rint(means[order]).astype(np.uint8)


# ---------- boundaries -> smooth vector polylines ----------

def trace_boundaries(labels, sigma, keep_every=3):
    """Chain the pixel-edge boundary graph into polylines that meet exactly at junctions, then smooth them."""
    h, w = labels.shape
    w1 = w + 1
    r, c = np.nonzero(labels[1:] != labels[:-1])
    a_h = (r + 1) * w1 + c
    r, c = np.nonzero(labels[:, 1:] != labels[:, :-1])
    a_v = r * w1 + c + 1
    a = np.concatenate([a_h, a_v])
    b = np.concatenate([a_h + 1, a_v + w1])
    n_edges = len(a)
    ends = np.concatenate([a, b])
    others = np.concatenate([b, a]).tolist()
    eids = np.tile(np.arange(n_edges), 2)
    order = np.argsort(ends, kind="stable")
    sorted_nodes = ends[order]
    nodes, start = np.unique(sorted_nodes, return_index=True)
    deg = np.diff(np.append(start, len(sorted_nodes)))
    incident = {}
    for node, s, d in zip(nodes.tolist(), start.tolist(), deg.tolist()):
        incident[node] = order[s:s + d].tolist()
    edge_ids = eids.tolist()
    used = [False] * n_edges

    def walk(node, slot):
        path = [node]
        while True:
            e = edge_ids[slot]
            used[e] = True
            node = others[slot]
            path.append(node)
            slots = incident[node]
            if len(slots) != 2:
                return path, False
            nxt = [s for s in slots if not used[edge_ids[s]]]
            if not nxt:
                return path, True
            slot = nxt[0]

    chains = []
    for node, slots in incident.items():
        if len(slots) != 2:
            for s in slots:
                if not used[edge_ids[s]]:
                    chains.append(walk(node, s))
    for node, slots in incident.items():
        if not used[edge_ids[slots[0]]]:
            chains.append(walk(node, slots[0]))

    out = []
    for path, closed in chains:
        pts = np.array([(p % w1, p // w1) for p in path], dtype=np.float64)
        out.append(_smooth(pts, closed, sigma, keep_every))
    return out


def _smooth(pts, closed, sigma, keep_every):
    if closed:
        ring = ndimage.gaussian_filter1d(pts[:-1], sigma, axis=0, mode="wrap")
        ring = ring[::keep_every]
        return np.vstack([ring, ring[:1]])
    sm = ndimage.gaussian_filter1d(pts, sigma, axis=0, mode="nearest")
    i = np.arange(len(pts))
    # Pin the ends so chains meeting at a junction or the wall edge still touch after smoothing.
    t = np.clip(np.minimum(i, len(pts) - 1 - i) / (2 * sigma), 0, 1)[:, None]
    sm = pts * (1 - t) + sm * t
    keep = np.unique(np.append(i[::keep_every], len(pts) - 1))
    return sm[keep]


def chunk_polylines(lines, size=40):
    """Split polylines into short pieces with bounding boxes so each sheet draws only what it shows."""
    chunks = []
    for line in lines:
        for s in range(0, max(len(line) - 1, 1), size - 1):
            chunks.append(line[s:s + size])
    boxes = np.array([[c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()] for c in chunks])
    return chunks, boxes


# ---------- layout ----------

def choose_layout(mural_w, mural_h, overlap):
    best = None
    for land in (True, False):
        pw, ph = (11.0, 8.5) if land else (8.5, 11.0)
        cw, ch = pw - 2 * MARGIN, ph - 2 * MARGIN - HEADER - FOOTER
        cols = max(1, math.ceil((mural_w - overlap) / (cw - overlap)))
        rows = max(1, math.ceil((mural_h - overlap) / (ch - overlap)))
        cand = Layout(land, pw, ph, cw, ch, overlap, cols, rows)
        if best is None or cand.count < best.count:
            best = cand
    return best


def row_name(r):
    name = ""
    r += 1
    while r:
        r, rem = divmod(r - 1, 26)
        name = chr(65 + rem) + name
    return name


def sheet_id(r, c):
    return f"{row_name(r)}{c + 1}"


def feet_inches(inches):
    ft, rem = divmod(round(inches), 12)
    return f"{ft} ft {rem} in"


# ---------- per-sheet number placement ----------

def place_numbers(crop, crop_regions, px_to_pt, style):
    """One number per same-color piece inside the crop, at its most interior point. Returns (labels, region ids labeled)."""
    dist = interior_distance(crop)
    placed, labeled = [], set()
    for k in np.unique(crop):
        comp, count = ndimage.label(crop == k)
        if not count:
            continue
        idx = np.arange(1, count + 1)
        radii = ndimage.maximum(dist, comp, idx)
        spots = ndimage.maximum_position(dist, comp, idx)
        text = str(k + 1)
        half_diag = math.hypot(stringWidth(text, style.font, 1), 0.72) / 2
        for radius, (y, x) in zip(radii, spots):
            # Always place a number so no region is left unlabeled. Tiny face details
            # may get type as small as 4.5 pt when they can't hold the usual minimum.
            raw = 0.85 * radius * px_to_pt / half_diag if radius > 0 else 4.5
            size = min(style.max_pt, max(4.5, raw))
            placed.append((x + 0.5, y + 0.5, text, size))
            labeled.add(int(crop_regions[y, x]))
    return placed, labeled


def complete_priority_labels(placed, labeled, crop, crop_regions, px_to_pt, style, masks):
    """Every region touching a priority mask (face/eye) gets a number on the sheet."""
    if not masks:
        return placed, labeled
    dist = interior_distance(crop)
    covered = np.zeros(crop.shape, dtype=bool)
    for m in masks:
        if m is not None:
            covered |= m >= 0.35
    if not np.any(covered):
        return placed, labeled
    placed = list(placed)
    labeled = set(labeled)
    for rid in np.unique(crop_regions[covered]):
        rid = int(rid)
        if rid in labeled or rid <= 0:
            continue
        m = crop_regions == rid
        if not np.any(m & covered):
            continue
        k = int(crop[m][0])
        text = str(k + 1)
        half_diag = math.hypot(stringWidth(text, style.font, 1), 0.72) / 2
        sub = np.where(m, dist, 0)
        cy, cx = np.unravel_index(sub.argmax(), sub.shape)
        radius = float(dist[cy, cx])
        raw = 0.85 * radius * px_to_pt / half_diag if radius > 0 else style.min_pt
        size = min(style.max_pt, max(4.5, raw))
        placed.append((cx + 0.5, cy + 0.5, text, size))
        labeled.add(rid)
    return placed, labeled


# ---------- PDF pages ----------

def text_color_for(rgb):
    return white if rgb_to_lab(np.array(rgb))[0] < 55 else HexColor("#000000")


def to_hex(rgb):
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


def fit_text(text, font, size, max_w):
    while size > 5 and stringWidth(text, font, size) > max_w:
        size -= 0.5
    return size


def draw_image_fit(cv, img, x, y, max_w, max_h):
    """Draw a PIL image scaled to fit the box, top-aligned. Returns (w, h, bottom y) in points."""
    s = min(max_w / img.width, max_h / img.height)
    w, h = img.width * s, img.height * s
    cv.drawImage(ImageReader(img), x + (max_w - w) / 2, y + max_h - h, w, h)
    return w, h, y + max_h - h


KIT_NOTES = [
    "1. Please only use a picture, logo, or artwork you have the right to use. This kit doesn't give you rights to anyone else's artwork.",
    "2. Paint amounts, colors, and coverage are estimates only. Real needs vary with brand, sheen, wall surface and texture,",
    "    and number of coats. This is not a supplier quote or a guarantee of coverage or color match.",
    "3. Provided as-is, for decorative painting guidance only. Not structural, safety, or professional contracting advice.",
    "4. Not affiliated with, endorsed by, or sponsored by xAI, SpaceXSI, or Grok.",
]


KEY_COLS = [("#", 0), ("Color", 26), ("Hex", 62), ("Sq ft", 118), ("Share", 160), ("Brand match (approximate)", 205)]
MIN_KEY_ROW_H = 12.0  # below this, numbers and names are too small to read; overflow to extra pages


def _notes_box_height():
    # title + notes + italic footer + padding, so the box never collides with the color key
    return 10 + 11 + 11 * len(KIT_NOTES) + 14 + 10


def draw_good_to_know(cv, x0, pw, box_bottom):
    """Draw the 'Good to know' notes in a box. Returns the top y of the box."""
    box_h = _notes_box_height()
    box_top = box_bottom + box_h
    width = pw - 2 * x0
    cv.setStrokeColor(HexColor("#555555"))
    cv.setFillColor(HexColor("#f4f4f4"))
    cv.setLineWidth(0.6)
    cv.roundRect(x0, box_bottom, width, box_h, 4, fill=1, stroke=1)
    cv.setFillColor(HexColor("#000000"))
    ny = box_top - 14
    cv.setFont("Helvetica-Bold", 8)
    cv.drawString(x0 + 8, ny, "Good to know")
    for note in KIT_NOTES:
        ny -= 11
        cv.setFont("Helvetica", fit_text(note, "Helvetica", 7.5, width - 16))
        cv.drawString(x0 + 8, ny, note)
    cv.setFont("Helvetica-Oblique", 7.5)
    cv.drawString(x0 + 8, box_bottom + 8,
                  "Colors are numbered lightest (1) to darkest. Screen colors and brand matches are approximate: "
                  "check physical paint chips before buying.")
    return box_top


def draw_key_header(cv, x0, yy):
    cv.setFillColor(HexColor("#000000"))
    cv.setFont("Helvetica-Bold", 9)
    for name, dx in KEY_COLS:
        cv.drawString(x0 + dx, yy, name)


def draw_key_row(cv, x0, yy, row_h, i, rgb, area, brand_map):
    font = min(9.0, max(7.0, row_h * 0.65))
    cv.setFillColor(HexColor(to_hex(rgb)))
    cv.rect(x0 + 26, yy - 2, 28, row_h * 0.8, fill=1, stroke=1)
    cv.setFillColor(HexColor("#000000"))
    cv.setFont("Helvetica-Bold", font)
    cv.drawString(x0, yy + 1, str(i + 1))
    cv.setFont("Helvetica", font)
    cv.drawString(x0 + 62, yy + 1, to_hex(rgb))
    cv.drawRightString(x0 + 150, yy + 1, f"{area['sqft']:.1f}")
    cv.drawRightString(x0 + 192, yy + 1, f"{100 * area['pixel_share']:.1f}%")
    match = brand_map.get(str(i + 1))
    if match:
        cv.setFillColor(HexColor(match["hex"]))
        cv.rect(x0 + 205, yy - 2, 14, row_h * 0.8, fill=1, stroke=1)
        cv.setFillColor(HexColor("#000000"))
        de = f", dE {match['delta_e']:.1f}" if "delta_e" in match else ""
        label = f"{match['name']} ({match['code']}{de}), approximate"
        cv.setFont("Helvetica", fit_text(label, "Helvetica", font, 360))
        cv.drawString(x0 + 224, yy + 1, label)
    else:
        cv.drawString(x0 + 205, yy + 1, "-")


def draw_color_key_pages(cv, job, palette, areas, brand_map):
    """Full-page color key, used when the cover has no room for a readable table. Returns page count."""
    pw, ph = letter
    x0, top = MARGIN * PT + 18, ph - MARGIN * PT - 24
    bottom = MARGIN * PT + 28
    row_h = 13.5
    per_page = max(1, int((top - 40 - bottom) / row_h) - 1)
    n, pages, i = len(palette), 0, 0
    while i < n:
        cv.setPageSize(letter)
        cv.setFillColor(HexColor("#000000"))
        title = "Color key" if pages == 0 else "Color key (continued)"
        cv.setFont("Helvetica-Bold", 16)
        cv.drawString(x0, top, title)
        cv.setFont("Helvetica", 9)
        brand = f"  ·  paint: {job['brand']}" if job.get("brand") else ""
        cv.drawString(x0, top - 16, f"{n} colors{brand}  ·  brand matches are the nearest published color, approximate")
        yy = top - 40
        draw_key_header(cv, x0, yy)
        end = min(n, i + per_page)
        for k in range(i, end):
            yy -= row_h
            draw_key_row(cv, x0, yy, row_h, k, palette[k], areas[k], brand_map)
        cv.setFont("Helvetica", 8)
        cv.setFillColor(HexColor("#000000"))
        footer = f"Colors {i + 1}–{end} of {n}"
        if end < n:
            footer += "  ·  continued on next page"
        cv.drawString(x0, MARGIN * PT + 10, footer)
        cv.showPage()
        pages += 1
        i = end
    return pages


def draw_cover(cv, job, preview, palette, areas, brand_map, layout):
    """Cover page. Returns 1. The color key follows on extra pages when it cannot fit readably here."""
    pw, ph = letter
    cv.setPageSize(letter)
    x0, top = MARGIN * PT + 18, ph - MARGIN * PT - 24
    cv.setFont("Helvetica-Bold", 22)
    cv.drawString(x0, top, "Paint-by-number mural")
    cv.setFont("Helvetica", 10)
    lines = [
        f"Wall {feet_inches(job['wall_w_in'])} wide x {feet_inches(job['wall_h_in'])} high "
        f"({job['wall_w_in']:.0f} x {job['wall_h_in']:.0f} in, {job['wall_sqft']:.1f} sq ft)  ·  "
        f"{len(palette)} colors  ·  {job['coats']} coat{'s' if job['coats'] > 1 else ''}  ·  {job['transfer']} transfer"
        + (f"  ·  paint: {job['brand']}" if job["brand"] else ""),
        f"{layout.count} sheets ({layout.cols} columns x {layout.rows} rows, "
        f"{'landscape' if layout.landscape else 'portrait'} letter, {job['overlap']:g} in overlap)  ·  "
        f"source image {job['fit_note']}",
    ]
    for i, line in enumerate(lines):
        cv.setFont("Helvetica", fit_text(line, "Helvetica", 9.5, pw - 2 * x0))
        cv.drawString(x0, top - 18 - 13 * i, line)

    y = top - 64
    if job["scale"] == 1:
        banner, fill = "TRUE SIZE: print every sheet at 100% / Actual size. Sheets are wall-size tracing templates.", HexColor("#1b7a3a")
    else:
        banner, fill = (f"REDUCED SCALE {job['scale'] * 100:g}%: 1 in on paper = {1 / job['scale']:g} in on the wall. "
                        "Not wall-size templates; reprint at --scale 1 to trace."), HexColor("#b3261e")
    cv.setFillColor(fill)
    cv.roundRect(x0, y - 6, pw - 2 * x0, 20, 4, fill=1, stroke=0)
    cv.setFillColor(white)
    cv.setFont("Helvetica-Bold", fit_text(banner, "Helvetica-Bold", 10, pw - 2 * x0 - 12))
    cv.drawString(x0 + 6, y, banner)

    _, _, bottom = draw_image_fit(cv, preview, x0, y - 16 - 4.0 * PT, pw - 2 * x0, 4.0 * PT)

    box_bottom = MARGIN * PT + 4
    box_top = draw_good_to_know(cv, x0, pw, box_bottom)

    # Legend table, only if every row stays readable; otherwise it goes on following pages.
    key_top = bottom - 22
    key_bottom = box_top + 12
    n = len(palette)
    row_h = (key_top - 14 - key_bottom) / (n + 1) if n else MIN_KEY_ROW_H
    key_on_cover = bool(n) and row_h >= MIN_KEY_ROW_H
    if key_on_cover:
        row_h = min(20, row_h)
        yy = key_top
        draw_key_header(cv, x0, yy)
        for i, rgb in enumerate(palette):
            yy -= row_h
            draw_key_row(cv, x0, yy, row_h, i, rgb, areas[i], brand_map)
    else:
        cv.setFillColor(HexColor("#000000"))
        cv.setFont("Helvetica-Oblique", 9)
        cv.drawString(x0, key_bottom + 6,
                      f"Color key for all {n} colors is on the following page{'s' if n > 40 else ''}.")
    cv.showPage()
    return 1, key_on_cover


INSTRUCTIONS = {
    "trace": "Transfer: scribble soft pencil over the back of the sheet (chalk on dark walls) or slip graphite "
             "transfer paper behind it, then trace every line and number with a ballpoint pen.",
    "template": "Transfer: cut along the bold outlines, tape the pieces back in place, and paint up to the cut edges "
                "(stencil style). Write each number on the wall before removing a piece.",
}


def draw_map(cv, job, preview, layout, mural_w, mural_h):
    page = landscape(letter) if mural_w >= mural_h else letter
    cv.setPageSize(page)
    pw, ph = page
    x0, top = MARGIN * PT + 30, ph - MARGIN * PT - 22
    cv.setFont("Helvetica-Bold", 16)
    cv.drawString(x0, top, "Placement map")
    cv.setFont("Helvetica", 9)
    cv.drawString(x0, top - 14, "Sheet A1's top-left corner goes at the wall's top-left corner. Rows run A, B, C... "
                                "top to bottom; columns 1, 2, 3... left to right.")
    box_w, box_h = pw - 2 * x0, ph - 2 * MARGIN * PT - 150
    s = min(box_w / mural_w, box_h / mural_h)          # points per paper inch on this map
    mw, mh = mural_w * s, mural_h * s
    mx, my = x0 + (box_w - mw) / 2, top - 40 - mh
    cv.drawImage(ImageReader(preview), mx, my, mw, mh)
    cv.setFillColor(Color(1, 1, 1, alpha=0.55))
    cv.rect(mx, my, mw, mh, fill=1, stroke=0)

    cv.setStrokeColor(INFO)
    cv.setLineWidth(0.9)
    cell_w, cell_h = layout.step_x * s, layout.step_y * s
    for c in range(1, layout.cols):
        cv.line(mx + c * cell_w, my, mx + c * cell_w, my + mh)
    for r in range(1, layout.rows):
        cv.line(mx, my + mh - r * cell_h, mx + mw, my + mh - r * cell_h)
    cv.setStrokeColor(HexColor("#000000"))
    cv.setLineWidth(1.2)
    cv.rect(mx, my, mw, mh)
    font = max(4.0, min(10.0, cell_w * 0.3, cell_h * 0.45))
    cv.setFillColor(HexColor("#000000"))
    cv.setFont("Helvetica-Bold", font)
    for r in range(layout.rows):
        for c in range(layout.cols):
            cx = mx + (c * layout.step_x + min((c + 1) * layout.step_x, mural_w)) / 2 * s
            cy = my + mh - (r * layout.step_y + min((r + 1) * layout.step_y, mural_h)) / 2 * s
            cv.drawCentredString(cx, cy - font * 0.35, sheet_id(r, c))

    cv.setFont("Helvetica", 9)
    cv.drawCentredString(mx + mw / 2, my + mh + 6,
                         f"wall width {job['wall_w_in']:.0f} in ({feet_inches(job['wall_w_in'])})")
    cv.saveState()
    cv.translate(mx - 8, my + mh / 2)
    cv.rotate(90)
    cv.drawCentredString(0, 0, f"wall height {job['wall_h_in']:.0f} in ({feet_inches(job['wall_h_in'])})")
    cv.restoreState()

    tips = [
        "1. Print every sheet at 100% / Actual size (no 'fit to page'). The 2 in scale bar on each sheet should measure 2 in.",
        "2. Start with A1 at the top-left corner; check that row A runs level. Work across each row, then down.",
        f"3. Overlap the shaded {job['overlap']:g} in bands so each pink crosshair sits exactly on the neighbor's crosshair; tape.",
        "4. " + INSTRUCTIONS[job["transfer"]],
        "5. Paint by number, lightest (1) to darkest, so darker colors cover any mistakes.",
    ]
    if job["scale"] != 1:
        tips[0] = (f"1. These sheets are at {job['scale'] * 100:g}% scale (1 in = {1 / job['scale']:g} in on the wall): "
                   "a reference set, not tracing templates.")
    yy = my - 22
    for tip in tips:
        cv.setFont("Helvetica", fit_text(tip, "Helvetica", 9, pw - 2 * x0))
        cv.drawString(x0, yy, tip)
        yy -= 13
    cv.showPage()


def crosshair(cv, x, y, r=0.11 * PT):
    cv.circle(x, y, r * 0.6, stroke=1, fill=0)
    cv.line(x - r, y, x + r, y)
    cv.line(x, y - r, x, y + r)


def neighbor_tag(cv, x, y, text, direction):
    """Small triangle pointing toward the neighbor, with its sheet ID."""
    t = 3.5
    tri = {"up": [(0, t), (-t, -t), (t, -t)], "down": [(0, -t), (-t, t), (t, t)],
           "left": [(-t, 0), (t, t), (t, -t)], "right": [(t, 0), (-t, t), (-t, -t)]}[direction]
    p = cv.beginPath()
    p.moveTo(x + tri[0][0], y + tri[0][1])
    for dx, dy in tri[1:]:
        p.lineTo(x + dx, y + dy)
    p.close()
    cv.drawPath(p, fill=1, stroke=0)
    cv.setFont("Helvetica-Bold", 7)
    if direction == "right":
        cv.drawRightString(x - 6, y - 2.5, text)
    else:
        cv.drawString(x + 6, y - 2.5, text)


def draw_sheet(cv, job, layout, r, c, mural_w, mural_h, chunks, boxes, labels, regions, palette, style,
               face_mask=None, eye_mask=None):
    page = (layout.page_w * PT, layout.page_h * PT)
    cv.setPageSize(page)
    k = job["scale"] / job["ppi"]                 # paper inches per work pixel
    sx0, sy0 = c * layout.step_x, r * layout.step_y
    left, bottom = MARGIN * PT, (MARGIN + FOOTER) * PT
    cw, ch, ov = layout.cw * PT, layout.ch * PT, layout.overlap * PT
    top = bottom + ch

    def to_page(x_px, y_px):
        return left + (x_px * k - sx0) * PT, top - (y_px * k - sy0) * PT

    wx0, wy0 = sx0 / k, sy0 / k
    wx1, wy1 = (sx0 + layout.cw) / k, (sy0 + layout.ch) / k
    tint = float(job.get("sheet_tint", 0))
    if tint > 0:
        x0i = max(0, int(math.floor(wx0)))
        y0i = max(0, int(math.floor(wy0)))
        x1i = min(labels.shape[1], int(math.ceil(wx1)))
        y1i = min(labels.shape[0], int(math.ceil(wy1)))
        if x1i > x0i and y1i > y0i:
            tile = palette[labels[y0i:y1i, x0i:x1i]]
            faded = np.clip(255.0 * (1 - tint) + tile.astype(np.float64) * tint, 0, 255).astype(np.uint8)
            buf = BytesIO()
            Image.fromarray(faded).save(buf, format="PNG")
            buf.seek(0)
            cv.drawImage(ImageReader(buf), left, bottom, cw, ch, preserveAspectRatio=False, anchor="sw")

    # Paper beyond the wall's right/bottom edge.
    wall_r = left + (mural_w - sx0) * PT
    wall_b = top - (mural_h - sy0) * PT
    cv.setFillColor(OFF_WALL)
    if wall_r < left + cw:
        cv.rect(wall_r, bottom, left + cw - wall_r, ch, fill=1, stroke=0)
    if wall_b > bottom:
        cv.rect(left, bottom, cw, wall_b - bottom, fill=1, stroke=0)

    # Overlap bands on top of tint so registration stays visible.
    cv.setFillColor(BAND)
    if c > 0:
        cv.rect(left, bottom, ov, ch, fill=1, stroke=0)
    if c < layout.cols - 1:
        cv.rect(left + cw - ov, bottom, ov, ch, fill=1, stroke=0)
    if r > 0:
        cv.rect(left, top - ov, cw, ov, fill=1, stroke=0)
    if r < layout.rows - 1:
        cv.rect(left, bottom, cw, ov, fill=1, stroke=0)

    sid = sheet_id(r, c)
    cv.setFillColor(HexColor("#dbe4f6"))
    cv.setFont("Helvetica-Bold", 40)
    cv.drawString(left + ov + 6, top - ov - 42, sid)

    # Region outlines (vector), clipped to the tile.
    cv.saveState()
    clip = cv.beginPath()
    clip.rect(left, bottom, cw, ch)
    cv.clipPath(clip, stroke=0, fill=0)
    hit = np.nonzero((boxes[:, 2] >= wx0) & (boxes[:, 0] <= wx1) & (boxes[:, 3] >= wy0) & (boxes[:, 1] <= wy1))[0]
    path = cv.beginPath()
    for i in hit:
        pts = chunks[i]
        path.moveTo(*to_page(*pts[0]))
        for x, y in pts[1:]:
            path.lineTo(*to_page(x, y))
    cv.setStrokeGray(style.line_gray)
    cv.setLineWidth(style.line_pt)
    cv.setLineJoin(1)
    cv.setLineCap(1)
    cv.drawPath(path, stroke=1, fill=0)
    cv.setStrokeColor(HexColor("#000000"))
    cv.setLineWidth(max(style.line_pt, 1.2))
    cv.rect(left - sx0 * PT, top - mural_h * PT + sy0 * PT, mural_w * PT, mural_h * PT, stroke=1, fill=0)
    cv.restoreState()

    # Color numbers go only in this sheet's map cell (tile minus the right/bottom bands). Later sheets are
    # taped over those bands, so every spot on the wall gets its number from exactly one visible sheet.
    x0p, y0p = int(math.floor(wx0)), int(math.floor(wy0))
    x1p, y1p = min(int(math.ceil(wx1)), labels.shape[1]), min(int(math.ceil(wy1)), labels.shape[0])
    crop = labels[y0p:y1p, x0p:x1p]
    cell_x1 = x1p if c == layout.cols - 1 else int(math.ceil((sx0 + layout.step_x) / k))
    cell_y1 = y1p if r == layout.rows - 1 else int(math.ceil((sy0 + layout.step_y) / k))
    cell_labels = labels[y0p:cell_y1, x0p:cell_x1]
    cell_regions = regions[y0p:cell_y1, x0p:cell_x1]
    placed, labeled = place_numbers(cell_labels, cell_regions, k * PT, style)
    masks = []
    if face_mask is not None:
        masks.append(face_mask[y0p:cell_y1, x0p:cell_x1])
    if eye_mask is not None:
        masks.append(eye_mask[y0p:cell_y1, x0p:cell_x1])
    placed, labeled = complete_priority_labels(placed, labeled, cell_labels, cell_regions, k * PT, style, masks)
    cv.setFillColor(HexColor("#000000"))
    for x, y, text, size in placed:
        px, py = to_page(x + x0p, y + y0p)
        cv.setFont(style.font, size)
        cv.drawCentredString(px, py - 0.36 * size, text)

    # Registration crosshairs at band centers: corners and edge midpoints, identical in wall coordinates on neighbors.
    cv.setStrokeColor(MARK)
    cv.setLineWidth(0.6)
    h = ov / 2 if ov else 0.15 * PT
    for x in (left + h, left + cw / 2, left + cw - h):
        for y in (bottom + h, bottom + ch / 2, top - h):
            if (x, y) != (left + cw / 2, bottom + ch / 2):
                crosshair(cv, x, y)

    cv.setFillColor(INFO)
    if r > 0:
        neighbor_tag(cv, left + cw / 2 + 40, top - h, sheet_id(r - 1, c), "up")
    if r < layout.rows - 1:
        neighbor_tag(cv, left + cw / 2 + 40, bottom + h, sheet_id(r + 1, c), "down")
    if c > 0:
        neighbor_tag(cv, left + h, bottom + ch / 2 + 30, sheet_id(r, c - 1), "left")
    if c < layout.cols - 1:
        neighbor_tag(cv, left + cw - h, bottom + ch / 2 + 30, sheet_id(r, c + 1), "right")

    # Header.
    cv.setFillColor(HexColor("#000000"))
    hy = top + 6
    wall_span = (f"wall x {sx0 / job['scale']:.1f}-{min(sx0 + layout.cw, mural_w) / job['scale']:.1f} in from left, "
                 f"y {sy0 / job['scale']:.1f}-{min(sy0 + layout.ch, mural_h) / job['scale']:.1f} in from top")
    head = f"Sheet {sid}   row {r + 1} of {layout.rows}, col {c + 1} of {layout.cols}   |   {wall_span}"
    right = "PRINT AT 100% / ACTUAL SIZE" if job["scale"] == 1 else f"{job['scale'] * 100:g}% SCALE REFERENCE"
    cv.setFont("Helvetica-Bold", 7.5)
    cv.drawRightString(left + cw, hy, right)
    cv.setFont("Helvetica-Bold", fit_text(head, "Helvetica-Bold", 8.5, cw - stringWidth(right, "Helvetica-Bold", 7.5) - 12))
    cv.drawString(left, hy, head)

    # Footer: colors on this sheet, scale bar, you-are-here grid. Everything stays below the tile.
    fy = MARGIN * PT + 2
    present = np.unique(crop)
    grid_h = FOOTER * PT - 6
    grid_w = min(grid_h * layout.cols / layout.rows, 1.2 * PT)
    grid_h = grid_w * layout.rows / layout.cols
    gx = left + cw - grid_w
    bar_x = gx - 2 * PT - 70
    cv.setFillColor(HexColor("#000000"))
    cv.setFont("Helvetica", 7)
    cv.drawString(left, fy + 18, "Colors on this sheet:")
    pitch = min(0.3 * PT, (bar_x - left - 20) / max(len(present), 1))
    for i, ci in enumerate(present):
        sx = left + i * pitch
        cv.setFillColor(HexColor(to_hex(palette[ci])))
        cv.setStrokeColor(HexColor("#000000"))
        cv.setLineWidth(0.4)
        cv.rect(sx, fy, pitch - 3, 14, fill=1, stroke=1)
        cv.setFillColor(text_color_for(palette[ci]))
        cv.setFont("Helvetica-Bold", 8)
        cv.drawCentredString(sx + (pitch - 3) / 2, fy + 4, str(ci + 1))

    cv.setStrokeColor(HexColor("#000000"))
    cv.setFillColor(HexColor("#000000"))
    cv.setLineWidth(0.8)
    cv.line(bar_x, fy + 4, bar_x + 2 * PT, fy + 4)
    for t in (0, 1, 2):
        cv.line(bar_x + t * PT, fy, bar_x + t * PT, fy + 8)
    cv.setFont("Helvetica", 7)
    label = "Scale check: this bar = 2.00 in"
    if job["scale"] != 1:
        label += f" (= {2 / job['scale']:g} in of wall)"
    cv.drawString(bar_x, fy + 12, label)

    cell_w, cell_h = grid_w / layout.cols, grid_h / layout.rows
    cv.setLineWidth(0.3)
    cv.setStrokeColor(HexColor("#888888"))
    cv.rect(gx, fy, grid_w, grid_h, fill=0, stroke=1)
    cv.setFillColor(MARK)
    cv.rect(gx + c * cell_w, fy + grid_h - (r + 1) * cell_h, cell_w, cell_h, fill=1, stroke=0)
    cv.setFillColor(HexColor("#000000"))
    cv.setFont("Helvetica", 6)
    cv.drawRightString(gx - 3, fy + 1, "you are here")
    cv.showPage()
    return labeled


# ---------- preview sampling & raster overview ----------

def _sheet_work_rect(layout, r, c, k):
    """Work-pixel axis-aligned bounds of a sheet tile (same as draw_sheet clipping)."""
    sx0, sy0 = c * layout.step_x, r * layout.step_y
    wx0, wy0 = sx0 / k, sy0 / k
    wx1, wy1 = (sx0 + layout.cw) / k, (sy0 + layout.ch) / k
    return wx0, wy0, wx1, wy1


def _rect_intersection_area(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _region_count_on_sheet(layout, r, c, k, regions):
    wx0, wy0, wx1, wy1 = _sheet_work_rect(layout, r, c, k)
    h, w = regions.shape
    x0, y0 = max(0, int(math.floor(wx0))), max(0, int(math.floor(wy0)))
    x1, y1 = min(w, int(math.ceil(wx1))), min(h, int(math.ceil(wy1)))
    if x1 <= x0 or y1 <= y0:
        return 0
    crop = regions[y0:y1, x0:x1]
    return len(np.unique(crop[crop > 0]))


def _face_mask_on_sheet(layout, r, c, k, face_mask):
    if face_mask is None:
        return 0.0
    wx0, wy0, wx1, wy1 = _sheet_work_rect(layout, r, c, k)
    h, w = face_mask.shape
    x0, y0 = max(0, int(math.floor(wx0))), max(0, int(math.floor(wy0)))
    x1, y1 = min(w, int(math.ceil(wx1))), min(h, int(math.ceil(wy1)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(face_mask[y0:y1, x0:x1].sum())


def _best_face_sheet(layout, face_mask, k, min_weight=1.0):
    if face_mask is None or float(face_mask.max()) < 0.1:
        return None
    best = (-1.0, 0, 0)
    for r in range(layout.rows):
        for c in range(layout.cols):
            area = _face_mask_on_sheet(layout, r, c, k, face_mask)
            if area > best[0]:
                best = (area, r, c)
    if best[0] < min_weight:
        return None
    return best[1], best[2], best[0]


def _best_region_sheet(layout, k, regions):
    best = (-1, 0, 0)
    for r in range(layout.rows):
        for c in range(layout.cols):
            n = _region_count_on_sheet(layout, r, c, k, regions)
            if n > best[0]:
                best = (n, r, c)
    return best[1], best[2], best[0]


def pick_preview_sheets(preview_sheets, layout, regions, face, k, face_mask=None):
    """Return [(r, c, reason), ...] in render order, at most preview_sheets entries."""
    if preview_sheets <= 0:
        return []
    picks = []
    seen = set()

    def add(r, c, reason):
        if (r, c) in seen:
            return
        seen.add((r, c))
        picks.append((r, c, reason))

    face_pick = _best_face_sheet(layout, face_mask, k) if face.get("found") else None
    if face_pick:
        r, c, area = face_pick
        add(r, c, f"most face-mask weight on sheet ({area:.1f})")
    else:
        r, c, n = _best_region_sheet(layout, k, regions)
        add(r, c, f"most regions ({n})")

    add(0, 0, "top-left corner sheet")

    if len(picks) < preview_sheets:
        ranked = []
        for r in range(layout.rows):
            for c in range(layout.cols):
                if (r, c) in seen:
                    continue
                ranked.append((-_region_count_on_sheet(layout, r, c, k, regions), r, c))
        ranked.sort()
        for _, r, c in ranked:
            if len(picks) >= preview_sheets:
                break
            add(r, c, f"regions ({_region_count_on_sheet(layout, r, c, k, regions)})")
    return picks[:preview_sheets]


def build_preview_kit_pdf(path, job, small, palette, areas, brand_map, layout, mural_w, mural_h,
                          chunks, boxes, labels, regions, palette_arr, style, sample_sheets,
                          face_mask=None, eye_mask=None):
    cv = Canvas(str(path), pagesize=letter, invariant=1, pageCompression=1)
    cv.setTitle("Paint-by-number mural (preview)")
    cover_pages, key_on_cover = draw_cover(cv, job, small, palette, areas, brand_map, layout)
    if not key_on_cover:
        draw_color_key_pages(cv, job, palette, areas, brand_map)
    draw_map(cv, job, small, layout, mural_w, mural_h)
    for r, c, _reason in sample_sheets:
        draw_sheet(cv, job, layout, r, c, mural_w, mural_h, chunks, boxes, labels, regions,
                   palette_arr, style, face_mask=face_mask, eye_mask=eye_mask)
    cv.save()
    return cover_pages


def rasterize_pdf_pages(pdf_path, out_dir, dpi=120):
    """Rasterize each PDF page to preview_pages/page-N.png; returns True if pages were written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("pdftoppm") is None:
        print("preview: pdftoppm not found; skipping rasterized preview_pages (install poppler-utils)")
        return False
    tmp = out_dir / "_ppm"
    tmp.mkdir(parents=True, exist_ok=True)
    prefix = tmp / "page"
    try:
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"preview: pdftoppm failed ({e.stderr or e}); skipping preview_pages")
        return False
    pages = sorted(tmp.glob("page-*.png"))
    for i, src in enumerate(pages, start=1):
        dest = out_dir / f"page-{i}.png"
        shutil.move(str(src), dest)
    for leftover in tmp.glob("*"):
        leftover.unlink(missing_ok=True)
    tmp.rmdir()
    return True


def _outline_font(style):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except OSError:
        return ImageFont.load_default()


def draw_preview_outlined(path, labels, regions, palette, chunks, layout, job, mural_w, mural_h, style):
    """Whole-mural PNG with fills, vector boundaries, numbers, and a faint sheet grid."""
    h, w = labels.shape
    k = job["scale"] / job["ppi"]
    base = Image.fromarray(palette[labels])
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    line_w = max(1, int(round(style.line_pt * job["ppi"] / 72 * 0.85)))
    for chunk in chunks:
        pts = [(float(x), float(y)) for x, y in chunk]
        draw.line(pts, fill=(0, 0, 0, 210), width=line_w, joint="curve")

    px_to_pt = k * PT * (w / max(mural_w * job["ppi"], 1))  # match sheet number sizing at mural width
    placed, _ = place_numbers(labels, regions, px_to_pt, style)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    for x, y, text, size in placed:
        pt = max(8, min(28, size * w / max(mural_w * job["ppi"], 1) * 0.55))
        try:
            font = ImageFont.truetype(font_path, int(round(pt)))
        except OSError:
            font = ImageFont.load_default()
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        draw.text((x - tw / 2, y - th / 2), text, fill=(0, 0, 0, 235), font=font)

    grid = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(grid)
    label_font = _outline_font(style)
    for r in range(layout.rows):
        for c in range(layout.cols):
            wx0, wy0, wx1, wy1 = _sheet_work_rect(layout, r, c, k)
            x0, y0 = max(0, int(math.floor(wx0))), max(0, int(math.floor(wy0)))
            x1, y1 = min(w, int(math.ceil(wx1))), min(h, int(math.ceil(wy1)))
            gdraw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(120, 120, 120, 140), width=1)
            sid = sheet_id(r, c)
            gdraw.text((x0 + 4, y0 + 4), sid, fill=(31, 79, 191, 180), font=label_font)

    out = Image.alpha_composite(base.convert("RGBA"), overlay)
    out = Image.alpha_composite(out, grid)
    out.convert("RGB").save(path, optimize=True)


# ---------- main ----------

def load_brand_map(path, n_colors):
    if not path:
        return {}
    data = json.loads(path.read_text())
    bad = [key for key in data if not (key.isdigit() and 1 <= int(key) <= n_colors)]
    if bad:
        raise SystemExit(f"--palette-json keys {bad} don't match colors 1..{n_colors}; rebuild it from this run's paint_areas.json")
    return data


def main():
    a = parse_args()
    if getattr(a, "_from_preview_dir", None):
        a.preview_only = False
    wall_w_in, wall_h_in = a.wall_width_ft * 12, a.wall_height_ft * 12
    style = STYLES[a.transfer]
    resolved = {"detail": a.detail, "prep": a.prep}
    crop_box = (0, 0, 0, 0)

    if getattr(a, "_from_preview_dir", None):
        meta = json.loads((a._from_preview_dir / PREVIEW_JOB_NAME).read_text())
        labels, palette, regions, face_mask, eye_mask = load_saved_labels(a._from_preview_dir)
        _, region_colors = label_regions(labels)
        ppi = float(meta["ppi"])
        fit_note = meta["fit_note"]
        crop_box = tuple(meta["crop_box"])
        face = dict(meta["face_priority"])
        face["mask"] = face_mask
        face["eye_mask"] = eye_mask
        resolved.update({"ppi": ppi, "fit_note": fit_note, "detail": meta.get("detail"), "prep": meta.get("prep")})
        print(f"from-preview: loaded {LABELS_ARCHIVE} from {a._from_preview_dir} "
              f"({labels.shape[1]}x{labels.shape[0]} work px, {int(regions.max())} regions)")
    else:
        src = ImageOps.exif_transpose(Image.open(a.image)).convert("RGB")
        src = apply_image_prep(src, a.prep)
        resolved["prep"] = a.prep
        want_upscale = False
        if not a.no_face_priority and (a.face_box or _detect_faces_yunet(src)):
            want_upscale = True
        ppi = min(TARGET_PPI, math.sqrt(MAX_WORK_PIXELS / (wall_w_in * wall_h_in)))
        ppi = _maybe_upscale_ppi(src, wall_w_in, wall_h_in, ppi, enabled=want_upscale)
        resolved["ppi"] = ppi
        img, fit_note, crop_box = fit_to_wall(src, wall_w_in, wall_h_in, ppi)
        resolved["fit_note"] = fit_note
        rgb = np.asarray(img)
        face = resolve_faces(src, img, crop_box, a)
        merge_min = lab_merge_min_dist(a.detail, a.colors, a.wall_width_ft, a.wall_height_ft)
        if face["mask"] is not None:
            labels = quantize_face_priority(
                img, a.colors, ppi, face["mask"],
                eye_mask=face.get("eye_mask"), n_eye=face.get("eye_colors_reserved") or 0,
                merge_min_dist=merge_min,
            )
            face_min = face["face_min_region_in"]
            eye_min = face.get("eye_min_region_in")
            win_bg = int(round(a.min_region_in / 4 * ppi)) | 1
            win_face = int(round(face_min / 4 * ppi)) | 1
            if a.detail == "photo" and is_canvas_job(a.wall_width_ft, a.wall_height_ft):
                win_bg = max(1, win_bg // 2)
            labels = majority_filter_face_aware(
                labels, a.colors, win_bg, win_face, face["mask"], eye_mask=face.get("eye_mask"),
                preserve_bg=(a.detail == "photo" and is_canvas_job(a.wall_width_ft, a.wall_height_ft)),
            )
            eye_kwargs = {}
            if eye_min:
                eye_kwargs = {
                    "eye_mask": face["eye_mask"],
                    "eye_area_px": (eye_min * ppi) ** 2,
                    "eye_thick_px": eye_min / 2 * ppi,
                }
            labels = clean_regions_face_aware(
                labels,
                (a.min_region_in * ppi) ** 2, a.min_region_in / 2 * ppi,
                (face_min * ppi) ** 2, face_min / 2 * ppi,
                face["mask"],
                **eye_kwargs,
            )
        else:
            labels = quantize(img, a.colors, ppi, merge_min_dist=merge_min)
            window = int(round(a.min_region_in / 4 * ppi)) | 1
            labels = majority_filter(labels, a.colors, window)
            labels = clean_regions(labels, (a.min_region_in * ppi) ** 2, a.min_region_in / 2 * ppi)
        labels, palette = finalize_palette(labels, rgb)
        regions, region_colors = label_regions(labels)
    n = len(palette)
    face_mask = face.get("mask")
    eye_mask = face.get("eye_mask")

    wall_sqft = wall_w_in * wall_h_in / 144
    counts = np.bincount(labels.ravel(), minlength=n)
    region_counts = np.bincount(region_colors[1:], minlength=n)
    brand_map = load_brand_map(a.palette_json, n)
    areas = []
    for i in range(n):
        entry = {"index": i + 1, "hex": to_hex(palette[i]), "pixel_share": round(counts[i] / labels.size, 5),
                 "sqft": round(counts[i] / labels.size * wall_sqft, 2), "regions": int(region_counts[i])}
        if str(i + 1) in brand_map:
            entry["brand_match"] = brand_map[str(i + 1)]
        areas.append(entry)

    mural_w, mural_h = wall_w_in * a.scale, wall_h_in * a.scale
    layout = choose_layout(mural_w, mural_h, a.overlap_in)
    job = {"wall_w_in": wall_w_in, "wall_h_in": wall_h_in, "wall_sqft": wall_sqft, "coats": a.coats,
           "transfer": a.transfer, "brand": a.brand, "overlap": a.overlap_in, "scale": a.scale, "ppi": ppi,
           "fit_note": fit_note, "sheet_tint": a.sheet_tint, "detail": a.detail}
    sigma = trace_sigma(ppi, a.min_region_in, face)

    a.out.mkdir(parents=True, exist_ok=True)
    preview = Image.fromarray(palette[labels])
    preview.save(a.out / "preview.png", optimize=True)
    small = preview.resize((min(1400, preview.width), round(min(1400, preview.width) * preview.height / preview.width)),
                           Image.LANCZOS)

    cv = Canvas(str(a.out / "mural.pdf"), pagesize=letter, invariant=1, pageCompression=1)
    cv.setTitle("Paint-by-number mural")
    cover_pages, key_on_cover = draw_cover(cv, job, small, palette, areas, brand_map, layout)
    key_pages = 0 if key_on_cover else draw_color_key_pages(cv, job, palette, areas, brand_map)
    draw_map(cv, job, small, layout, mural_w, mural_h)
    front_pages = cover_pages + key_pages + 1  # cover + optional key pages + placement map
    labeled = set()
    chunks, boxes = chunk_polylines(trace_boundaries(labels, sigma=sigma))
    f_mask, e_mask = face_mask, eye_mask
    if not a.preview_only:
        for r in range(layout.rows):
            for c in range(layout.cols):
                labeled |= draw_sheet(cv, job, layout, r, c, mural_w, mural_h, chunks, boxes, labels, regions,
                                      palette, style, face_mask=f_mask, eye_mask=e_mask)
    cv.save()

    preview_samples = []
    if a.preview_only:
        save_preview_job(a.out, a, resolved, face, labels, palette, regions, crop_box, face_mask, eye_mask)
        print(f"preview job: {a.out / PREVIEW_JOB_NAME}")
        print(f"preview job: {a.out / LABELS_ARCHIVE}")
        k_sheet = job["scale"] / job["ppi"]
        preview_samples = pick_preview_sheets(a.preview_sheets, layout, regions, face, k_sheet, face_mask)
        kit_pdf = a.out / "preview_kit.pdf"
        build_preview_kit_pdf(kit_pdf, job, small, palette, areas, brand_map, layout, mural_w, mural_h,
                              chunks, boxes, labels, regions, palette, style, preview_samples,
                              face_mask=f_mask, eye_mask=e_mask)
        outlined = a.out / "preview_outlined.png"
        draw_preview_outlined(outlined, labels, regions, palette, chunks, layout, job, mural_w, mural_h, style)
        pages_dir = a.out / "preview_pages"
        rasterize_pdf_pages(kit_pdf, pages_dir)
        print(f"preview: {a.out / 'preview.png'}")
        print(f"preview: {outlined}")
        print(f"preview: {kit_pdf}")
        if pages_dir.is_dir() and any(pages_dir.glob("page-*.png")):
            print(f"preview: {pages_dir}/page-*.png")
        for r, c, reason in preview_samples:
            print(f"preview sheet {sheet_id(r, c)} (row {r + 1}, col {c + 1}): {reason}")
        print(f"full build will use {layout.count} sheets ({layout.cols}x{layout.rows} "
              f"{'landscape' if layout.landscape else 'portrait'})")

    summary = {
        "wall_width_ft": a.wall_width_ft, "wall_height_ft": a.wall_height_ft, "wall_sqft": round(wall_sqft, 2),
        "coats": a.coats, "coverage_sqft_per_gal": a.coverage_sqft_per_gal, "brand": a.brand,
        "requested_colors": a.colors, "colors_used": n, "transfer": a.transfer, "scale": a.scale,
        "fit": fit_note, "work_px_per_wall_in": round(ppi, 2), "min_region_in": a.min_region_in,
        "detail": a.detail, "prep": resolved.get("prep"), "trace_sigma": round(sigma, 3),
        "label_metrics": label_metrics(labels, regions, face_mask, eye_mask),
        "sheets": {"count": layout.count, "rows": layout.rows, "cols": layout.cols,
                   "orientation": "landscape" if layout.landscape else "portrait", "overlap_in": a.overlap_in},
        "regions": int(regions.max()),
        "front_pages": front_pages,
        "colors": areas,
        "face_priority": {
            "enabled": face["enabled"],
            "found": face["found"],
            "detector": face["detector"],
            "faces": face["faces"],
            "face_colors_reserved": face["face_colors_reserved"],
            "face_min_region_in": face["face_min_region_in"],
            "eye_priority": face.get("eye_count", 0) > 0,
            "eye_count": face.get("eye_count", 0),
            "eye_colors_reserved": face.get("eye_colors_reserved", 0),
            "eye_min_region_in": face.get("eye_min_region_in"),
            "eye_disks_work": face.get("eye_disks_work", []),
        },
    }
    if not a.preview_only:
        summary["unlabeled_regions"] = int(regions.max()) - len(labeled)
        if getattr(a, "_from_preview_dir", None):
            saved_lm = json.loads((a._from_preview_dir / PREVIEW_JOB_NAME).read_text()).get("label_metrics", {})
            live = summary["label_metrics"]
            summary["label_parity"] = {
                "regions_match": saved_lm.get("regions") == live["regions"],
                "face_regions_match": saved_lm.get("face_regions") == live.get("face_regions"),
            }
    (a.out / "paint_areas.json").write_text(json.dumps(summary, indent=2) + "\n")
    pdf_mb = (a.out / "mural.pdf").stat().st_size / 1e6
    sheet_note = f"{layout.count} sheets" if a.preview_only else f"{summary['sheets']['count']} sheets"
    print(f"{n} colors ({a.colors} requested), {summary['regions']} regions, "
          f"{sheet_note} {layout.cols}x{layout.rows} {summary['sheets']['orientation']}, "
          f"{pdf_mb:.1f} MB -> {a.out / 'mural.pdf'}")
    if summary.get("unlabeled_regions"):
        print(f"warning: {summary['unlabeled_regions']} regions got no number on any sheet")


if __name__ == "__main__":
    main()
