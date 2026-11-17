#!/usr/bin/env python3
"""Paint shopping list from a mural's paint_areas.json.

Per color: gallons = sq ft x coats / coverage x (1 + waste), then the cheapest sensible mix of
real can sizes. Writes <prefix>.md, <prefix>.csv and <prefix>.pdf; with --merge-pdf it also writes
a copy of the mural PDF with the shopping list inserted after the placement map (after paint_areas.json front_pages, default page 3).
"""
import argparse
import csv
import json
import math
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

CAN_GAL = {"sample": 1 / 16, "quart": 0.25, "gallon": 1.0}   # 8 fl oz sample jar = 1/16 gal
# Rough US big-box prices (Behr Premium Plus at Home Depot, Oct 2026). Pass current prices with --prices.
DEFAULT_PRICES = "sample=6.98,quart=18.48,gallon=34.98"
# Pay up to this much more to avoid buying a lot of extra tinted paint, which can't be returned.
LESS_PAINT_PREMIUM = 0.10
# Past two jars (16 oz) a can is cheaper and keeps one color in one consistent container.
MAX_SAMPLE_JARS = 2


def parse_prices(text):
    prices = {}
    for part in text.split(","):
        size, _, price = part.partition("=")
        size = size.strip()
        if size not in CAN_GAL:
            raise argparse.ArgumentTypeError(f"unknown can size {size!r}; use {', '.join(CAN_GAL)}")
        prices[size] = float(price)
    if not prices:
        raise argparse.ArgumentTypeError("give at least one can size price")
    return prices


def gallons_needed(sqft, coats, coverage, waste):
    return sqft * coats / coverage * (1 + waste)


def best_purchase(need, prices):
    """Cheapest can mix covering `need` gallons; among mixes within LESS_PAINT_PREMIUM of it, the least paint.

    Sample jars (at most MAX_SAMPLE_JARS) are never mixed with cans, so a color isn't split across products.
    Returns {size: count}.
    """
    if need <= 0:
        return {}
    options = []
    jars = math.ceil(need / CAN_GAL["sample"] - 1e-9)
    cans_priced = "quart" in prices or "gallon" in prices
    if "sample" in prices and (jars <= MAX_SAMPLE_JARS or not cans_priced):
        options.append({"sample": jars})
    g_max = math.ceil(need) if "gallon" in prices else 0
    q_max = math.ceil(need / CAN_GAL["quart"]) if "quart" in prices else 0
    for g in range(g_max + 1):
        for q in range(q_max + 1):
            if g + q and g * CAN_GAL["gallon"] + q * CAN_GAL["quart"] >= need - 1e-9:
                options.append({k: v for k, v in (("gallon", g), ("quart", q)) if v})

    def cost(o):
        return sum(prices[k] * v for k, v in o.items())

    def volume(o):
        return sum(CAN_GAL[k] * v for k, v in o.items())

    cheapest = min(cost(o) for o in options)
    near = [o for o in options if cost(o) <= cheapest * (1 + LESS_PAINT_PREMIUM) + 1e-9]
    return min(near, key=lambda o: (volume(o), sum(o.values()), cost(o)))


def describe(buy):
    names = {"gallon": ("gallon", "gallons"), "quart": ("quart", "quarts"),
             "sample": ("sample jar (8 oz)", "sample jars (8 oz)")}
    return " + ".join(f"{n} {names[k][n != 1]}" for k, n in sorted(buy.items(), key=lambda kv: -CAN_GAL[kv[0]]))


def smallest_purchase(prices):
    """One unit of the smallest priced container size."""
    for size in ("sample", "quart", "gallon"):
        if size in prices:
            return {size: 1}
    return {}


def color_has_area(c, wall_sqft):
    if c.get("pixel_share", 0) > 0:
        return True
    if c.get("sqft", 0) > 0:
        return True
    return effective_sqft(c, wall_sqft) > 0


def effective_sqft(c, wall_sqft):
    sq = float(c.get("sqft") or 0)
    if sq > 0:
        return sq
    share = float(c.get("pixel_share") or 0)
    if share > 0 and wall_sqft:
        return share * wall_sqft
    return 0.0


def format_area(sqft):
    if sqft <= 0:
        return "0"
    if sqft < 1.0:
        return f"{sqft * 144:.1f} sq in"
    return f"{sqft:.1f} sq ft"


def format_total_area(total_sqft):
    if total_sqft < 1.0:
        return f"{total_sqft * 144:.1f} sq in"
    return f"{total_sqft:.1f} sq ft"


def job_area_header(areas):
    sq = float(areas.get("wall_sqft") or 0)
    wf, wh = areas.get("wall_width_ft"), areas.get("wall_height_ft")
    if wf is not None and wh is not None and sq < 2.0:
        w_in, h_in = wf * 12, wh * 12

        def fmt_in(x):
            r = round(x)
            return str(int(r)) if abs(x - r) < 0.05 else f"{x:.1f}"

        return f"{fmt_in(w_in)} x {fmt_in(h_in)} in canvas (~{sq:.2f} sq ft)"
    return f"{sq:.1f} sq ft wall"


def build_rows(areas, coats, coverage, waste, prices):
    wall_sqft = float(areas.get("wall_sqft") or 0)
    rows = []
    for c in areas["colors"]:
        sqft = effective_sqft(c, wall_sqft)
        need = gallons_needed(sqft, coats, coverage, waste)
        buy = best_purchase(need, prices)
        if color_has_area(c, wall_sqft) and not buy:
            buy = smallest_purchase(prices)
        match = c.get("brand_match")
        rows.append({
            "color": c["index"], "hex": c["hex"],
            "brand_match": f"{match['name']} ({match['code']}), approximate" if match else "",
            "sqft": sqft,
            "area_display": format_area(sqft),
            "gallons_needed": round(need, 3), "fl_oz_needed": round(need * 128, 1),
            "buy": describe(buy), "est_cost": round(sum(prices[k] * v for k, v in buy.items()), 2),
        })
    return rows


def write_md(path, rows, header):
    lines = ["# Paint shopping list", "", header, "",
             "| # | Color | Brand match | Area | Gal needed | Fl oz | Buy | Est. cost |",
             "|---|---|---|---:|---:|---:|---|---:|"]
    for r in rows:
        lines.append(f"| {r['color']} | {r['hex']} | {r['brand_match'] or '-'} | {r['area_display']} | "
                     f"{r['gallons_needed']:.2f} | {r['fl_oz_needed']:.1f} | {r['buy']} | ${r['est_cost']:.2f} |")
    total_sq = sum(r["sqft"] for r in rows)
    lines.append(f"| | | **Total** | **{format_total_area(total_sq)}** | "
                 f"**{sum(r['gallons_needed'] for r in rows):.2f}** | **{sum(r['fl_oz_needed'] for r in rows):.0f}** | | **${sum(r['est_cost'] for r in rows):.2f}** |")
    path.write_text("\n".join(lines) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


MIN_LIST_ROW_H = 13.0
NOTE_SIZE = 8
NOTE_LEADING = 11


def wrap_text(text, font, size, width):
    words = text.split()
    if not words:
        return [""]
    lines, cur = [], words[0]
    for w in words[1:]:
        trial = f"{cur} {w}"
        if stringWidth(trial, font, size) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def write_pdf(path, rows, header, notes):
    """Write a readable shopping list, flowing onto extra pages when the table is long."""
    pw, ph = letter
    cv = Canvas(str(path), pagesize=letter, invariant=1)
    x0, top = 50, ph - 50
    width = pw - 2 * x0
    cols = [("#", 0), ("Color", 18), ("Brand match", 62), ("Area", 282), ("Gal", 322), ("Fl oz", 360), ("Buy", 375),
            ("Est.", 512)]
    row_h = MIN_LIST_ROW_H
    size = 8.0
    header_lines = wrap_text(header, "Helvetica", 8, width)
    note_blocks = [wrap_text(note, "Helvetica", NOTE_SIZE, width) for note in notes]
    notes_h = 16 + sum(len(b) * NOTE_LEADING for b in note_blocks) + 8
    totals_h = row_h + 16
    n = len(rows)
    pages_needed = 1
    # First page has title + header lines; later pages have a shorter title.
    def table_top(page):
        y = top
        y -= 24 if page == 0 else 20  # title
        y -= 12 * len(header_lines) + 8
        y -= 14  # column headers
        return y

    # Decide how many data rows fit on each page, reserving totals+notes on the last page.
    remaining = n
    page_slices = []
    page = 0
    while remaining > 0 or not page_slices:
        y0 = table_top(page)
        # If this might be the last page, try to fit remaining rows + totals + notes.
        take = max(1, int((y0 - 40) / row_h)) if remaining else 0
        if remaining <= take:
            # Check if totals+notes also fit under the remaining rows.
            y_after = y0 - remaining * row_h - totals_h - notes_h
            if y_after >= 36 or remaining == 0:
                page_slices.append((n - remaining, n, True))
                remaining = 0
            else:
                # Leave these rows for a final page that only has a few rows + notes,
                # or put as many as will fit above a "continued" footer.
                room_for_all = int((y0 - totals_h - notes_h - 36) / row_h)
                if room_for_all >= 1 and remaining <= take:
                    # not enough for everything; fill this page without totals
                    fit = max(1, int((y0 - 40) / row_h))
                    fit = min(fit, remaining - 1) if remaining > 1 else remaining
                    if fit >= remaining:
                        fit = remaining - 1 if remaining > 1 else remaining
                    start = n - remaining
                    page_slices.append((start, start + fit, False))
                    remaining -= fit
                else:
                    fit = min(remaining, max(1, int((y0 - 40) / row_h)))
                    # if putting all remaining here would crush notes, hold at least 1 for next page
                    if remaining == fit and (y0 - fit * row_h - totals_h - notes_h) < 36 and remaining > 1:
                        fit = remaining - 1
                    start = n - remaining
                    last = (start + fit == n)
                    page_slices.append((start, start + fit, last))
                    remaining -= fit
        else:
            start = n - remaining
            page_slices.append((start, start + take, False))
            remaining -= take
        page += 1
        if page > 50:
            raise RuntimeError("shopping list pagination failed")

    def draw_row(cv, r, y):
        cv.setFillColor(HexColor(r["hex"]))
        cv.rect(x0 + 18, y - 2, 36, row_h * 0.8, fill=1, stroke=1)
        cv.setFillColor(HexColor("#000000"))
        cv.setFont("Helvetica-Bold", size)
        cv.drawString(x0, y + 1, str(r["color"]))
        cv.setFont("Helvetica", size)
        brand = r["brand_match"] or "-"
        while brand and stringWidth(brand, "Helvetica", size) > 205:
            brand = brand[:-1]
        cv.drawString(x0 + 62, y + 1, brand)
        cv.drawRightString(x0 + 282, y + 1, r["area_display"])
        cv.drawRightString(x0 + 322, y + 1, f"{r['gallons_needed']:.2f}")
        cv.drawRightString(x0 + 360, y + 1, f"{r['fl_oz_needed']:.1f}")
        cv.drawString(x0 + 375, y + 1, r["buy"])
        cv.drawRightString(x0 + 512, y + 1, f"${r['est_cost']:.2f}")

    for p_i, (start, end, is_last) in enumerate(page_slices):
        y = top
        cv.setFillColor(HexColor("#000000"))
        cv.setFont("Helvetica-Bold", 18 if p_i == 0 else 14)
        title = "Paint shopping list" if p_i == 0 else "Paint shopping list (continued)"
        cv.drawString(x0, y, title)
        y -= 24 if p_i == 0 else 20
        cv.setFont("Helvetica", 8)
        for line in header_lines:
            cv.drawString(x0, y, line)
            y -= 12
        y -= 8
        cv.setFont("Helvetica-Bold", 9)
        for name, dx in cols:
            if name in ("Area", "Gal", "Fl oz", "Est."):
                cv.drawRightString(x0 + dx, y, name)
            else:
                cv.drawString(x0 + dx, y, name)
        for r in rows[start:end]:
            y -= row_h
            draw_row(cv, r, y)
        if is_last:
            y -= row_h + 4
            cv.setStrokeColor(HexColor("#000000"))
            cv.setLineWidth(0.8)
            cv.line(x0, y + row_h * 0.6, x0 + 512, y + row_h * 0.6)
            cv.setFillColor(HexColor("#000000"))
            cv.setFont("Helvetica-Bold", 9)
            cv.drawString(x0 + 62, y, "Total")
            cv.drawRightString(x0 + 282, y, format_total_area(sum(r["sqft"] for r in rows)))
            cv.drawRightString(x0 + 322, y, f"{sum(r['gallons_needed'] for r in rows):.2f}")
            cv.drawRightString(x0 + 360, y, f"{sum(r['fl_oz_needed'] for r in rows):.0f}")
            cv.drawRightString(x0 + 512, y, f"${sum(r['est_cost'] for r in rows):.2f}")
            y -= 20
            for block in note_blocks:
                for line in block:
                    cv.setFont("Helvetica", NOTE_SIZE)
                    cv.setFillColor(HexColor("#000000"))
                    cv.drawString(x0, y, line)
                    y -= NOTE_LEADING
                y -= 2
        else:
            cv.setFont("Helvetica", 8)
            cv.setFillColor(HexColor("#000000"))
            cv.drawString(x0, 36, f"Colors {start + 1}–{end} of {n}  ·  continued on next page")
        cv.showPage()
    cv.save()


def merge(mural_pdf, list_pdf, out_pdf, after_page=2):
    writer = PdfWriter()
    mural = PdfReader(mural_pdf)
    for i, page in enumerate(mural.pages):
        writer.add_page(page)
        if i == after_page - 1:
            for p in PdfReader(list_pdf).pages:
                writer.add_page(p)
    with open(out_pdf, "wb") as f:
        writer.write(f)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--areas", required=True, type=Path, help="paint_areas.json from mural_tiler.py")
    p.add_argument("--coats", type=int, choices=(1, 2), help="default: the value recorded in paint_areas.json")
    p.add_argument("--coverage-sqft-per-gal", type=float, help="per coat; default: the value in paint_areas.json")
    p.add_argument("--waste", type=float, default=0.10, help="extra fraction for spills, brushes and touch-ups")
    p.add_argument("--prices", type=parse_prices, default=parse_prices(DEFAULT_PRICES),
                   help=f"can prices, e.g. {DEFAULT_PRICES!r}; leave out 'sample' to never buy sample jars")
    p.add_argument("--out-prefix", type=Path, help="default: shopping_list next to --areas")
    p.add_argument("--merge-pdf", type=Path, help="mural.pdf to copy with the shopping list inserted after the placement map")
    p.add_argument("--merged-out", type=Path, help="where to write the merged PDF (required with --merge-pdf)")
    p.add_argument("--extra-note", action="append", default=[], dest="extra_notes",
                   help="extra note line on the shopping list (repeatable)")
    a = p.parse_args()
    if bool(a.merge_pdf) != bool(a.merged_out):
        p.error("--merge-pdf and --merged-out go together")

    areas = json.loads(a.areas.read_text())
    coats = a.coats or areas["coats"]
    coverage = a.coverage_sqft_per_gal or areas["coverage_sqft_per_gal"]
    rows = build_rows(areas, coats, coverage, a.waste, a.prices)
    price_text = ", ".join(f"{k} ${v:.2f}" for k, v in sorted(a.prices.items(), key=lambda kv: CAN_GAL[kv[0]]))
    brand = f"{areas['brand']} · " if areas.get("brand") else ""
    header = (f"{brand}{job_area_header(areas)} · {coats} coat{'s' if coats > 1 else ''} · "
              f"{coverage:g} sq ft/gal per coat · +{a.waste * 100:g}% waste · prices: {price_text}")
    notes = [
        "Gallons = sq ft x coats / coverage x (1 + waste). Buys: the cheapest mix of cans, or the smallest one within 10% of it.",
        "Prices are estimates; check the store. Paint is usually non-returnable, and tinted colors can vary between batches.",
        "8 oz samples: many brands label them for color testing only. Use quarts outdoors, on high-traffic walls, or when unsure.",
        "Brand matches are the nearest published color to the screen color: confirm with physical paint chips before buying.",
    ]
    notes.extend(a.extra_notes)

    prefix = a.out_prefix or a.areas.with_name("shopping_list")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_md(prefix.with_suffix(".md"), rows, header)
    write_csv(prefix.with_suffix(".csv"), rows)
    write_pdf(prefix.with_suffix(".pdf"), rows, header, notes)
    if a.merge_pdf:
        after = int(areas.get("front_pages") or 2)
        merge(a.merge_pdf, prefix.with_suffix(".pdf"), a.merged_out, after_page=after)
    print(prefix.with_suffix(".md").read_text())


if __name__ == "__main__":
    main()
