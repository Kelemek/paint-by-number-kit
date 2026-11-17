#!/usr/bin/env python3
"""Match each mural color to the nearest color in a brand palette (CIEDE2000). Matches are approximate.

Brand palette JSON:  {"brand": "...", "colors": [{"name": "...", "code": "...", "hex": "#rrggbb"}, ...]}
Output (--out), ready for mural_tiler.py --palette-json:
  {"1": {"name": "...", "code": "...", "hex": "#rrggbb", "delta_e": 2.3}, ...}
dE under ~2 is hard to tell apart; over ~5 is a clearly different color, so check chips.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def hex_to_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def rgb_to_lab(rgb):
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def ciede2000(lab1, lab2):
    """CIEDE2000 color difference; broadcasts over leading dimensions (Sharma, Wu & Dalal 2005)."""
    L1, a1, b1 = np.moveaxis(lab1, -1, 0)
    L2, a2, b2 = np.moveaxis(lab2, -1, 0)
    c_bar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(c_bar ** 7 / (c_bar ** 7 + 25.0 ** 7)))
    a1p, a2p = a1 * (1 + g), a2 * (1 + g)
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dL, dC = L2 - L1, c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(c1p * c2p == 0, 0, dh)
    dH = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2))
    L_bar, c_bar_p = (L1 + L2) / 2, (c1p + c2p) / 2
    h_sum = h1p + h2p
    h_bar = np.where(c1p * c2p == 0, h_sum,
                     np.where(np.abs(h1p - h2p) <= 180, h_sum / 2,
                              np.where(h_sum < 360, (h_sum + 360) / 2, (h_sum - 360) / 2)))
    t = (1 - 0.17 * np.cos(np.radians(h_bar - 30)) + 0.24 * np.cos(np.radians(2 * h_bar))
         + 0.32 * np.cos(np.radians(3 * h_bar + 6)) - 0.20 * np.cos(np.radians(4 * h_bar - 63)))
    s_l = 1 + 0.015 * (L_bar - 50) ** 2 / np.sqrt(20 + (L_bar - 50) ** 2)
    s_c = 1 + 0.045 * c_bar_p
    s_h = 1 + 0.015 * c_bar_p * t
    r_t = (-2 * np.sqrt(c_bar_p ** 7 / (c_bar_p ** 7 + 25.0 ** 7))
           * np.sin(np.radians(60 * np.exp(-(((h_bar - 275) / 25) ** 2)))))
    return np.sqrt((dL / s_l) ** 2 + (dC / s_c) ** 2 + (dH / s_h) ** 2 + r_t * (dC / s_c) * (dH / s_h))


def match(hexes, palette):
    ours = rgb_to_lab([hex_to_rgb(h) for h in hexes])
    theirs = rgb_to_lab([hex_to_rgb(c["hex"]) for c in palette])
    de = ciede2000(ours[:, None, :], theirs[None, :, :])
    best = de.argmin(axis=1)
    return [(palette[j], float(de[i, j])) for i, j in enumerate(best)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--areas", required=True, type=Path, help="paint_areas.json from mural_tiler.py")
    p.add_argument("--palette", required=True, type=Path, help="brand palette JSON")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()

    colors = json.loads(a.areas.read_text())["colors"]
    brand = json.loads(a.palette.read_text())
    if brand.get("example_data"):
        print("warning: this palette is EXAMPLE DATA, not real brand codes")
    result = {}
    for c, (m, de) in zip(colors, match([c["hex"] for c in colors], brand["colors"])):
        result[str(c["index"])] = {"name": m["name"], "code": m["code"], "hex": m["hex"], "delta_e": round(de, 1)}
        print(f"{c['index']:>3}  {c['hex']}  ->  {m['code']:<12} {m['name']:<28} {m['hex']}  dE {de:4.1f}  (approximate)")
    a.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
