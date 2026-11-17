# Paint-by-number mural kit

Turns one picture into a printable kit a group can paint together. You get numbered 8.5x11 sheets
that tape up on the wall, a placement map, and a paint shopping list. Everything runs locally and the
same inputs always produce the same files.

## Install

Needs Python 3.10+ and poppler-utils (`pdftoppm`/`pdfinfo`/`pdftotext`, used only for checking output).

```bash
python3 -m pip install --user --break-system-packages -r ~/mural-kit/requirements.txt
```

Dependency versions in `requirements.txt` are pinned on purpose for safety and are bumped deliberately.
Tested on Python 3.13 with Pillow 13.0.0, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, reportlab 5.1.0, pypdf 6.21.0, and opencv-python-headless 5.0.0.93.
Drop `--break-system-packages` on systems without PEP 668 (or use a venv).

## Files

| File | What it does |
|---|---|
| `mural_tiler.py` | picture → `mural.pdf` (cover, placement map, sheets), `paint_areas.json`, `preview.png` |
| `brand_match.py` | nearest brand color per mural color (CIEDE2000) → palette map for `--palette-json` |
| `paint_math.py` | `paint_areas.json` → shopping list (`.md`, `.csv`, `.pdf`), optionally merged into the kit PDF |
| `example_brand_palette.json` | **example data** showing the brand palette format (made-up names and codes) |
| `sample/run_sample.sh` | builds the sample kits end to end |
| `sample/verify_kit.py` | checks sheet IDs against the map and overlap-band registration between neighbors |

## Workflow

```bash
ARGS=(--image photo.jpg --wall-width-ft 10 --wall-height-ft 8 --colors 8 --coats 2 --transfer trace --brand "Behr")

# 1. Quick preview (cover + map + areas, no sheets) to show the organizer.
python3 mural_tiler.py "${ARGS[@]}" --preview-only --out job

# 2. Match colors to the chosen brand (palette built from that brand's published color list; see below).
python3 brand_match.py --areas job/paint_areas.json --palette behr_palette.json --out job/palette_map.json

# 3. Full kit with brand labels.
python3 mural_tiler.py "${ARGS[@]}" --palette-json job/palette_map.json --out job

# 4. Shopping list, inserted as page 3 of a copy of the kit.
python3 paint_math.py --areas job/paint_areas.json --prices "sample=6.98,quart=18.48,gallon=34.98" \
  --merge-pdf job/mural.pdf --merged-out job/mural_kit.pdf
```

Step 3 must use the same image and settings as step 1; color numbers are only stable for identical
inputs. The tiler rejects a palette map whose numbers don't fit the run.

### mural_tiler.py flags

| Flag | Default | Meaning |
|---|---|---|
| `--image` | required | any Pillow-readable picture (EXIF rotation is applied) |
| `--wall-width-ft`, `--wall-height-ft` | required | wall size |
| `--colors N` | required | paint colors; 6-12 is easiest for groups. Fewer may be used if cleanup merges a color away (the summary says so) |
| `--coats 1\|2` | 2 | recorded in `paint_areas.json` for `paint_math.py` and shown on the cover |
| `--coverage-sqft-per-gal` | 375 | per coat; recorded for `paint_math.py` |
| `--overlap-in` | 0.5 | overlap between neighboring sheets, paper inches |
| `--min-region-in` | 3.0 | wall inches. Regions under this squared in area, or thinner than half of it, merge into the neighbor color they touch most. Raise it for younger or bigger groups |
| `--transfer trace\|template` | trace | `trace`: thin 0.6 pt lines, 7-14 pt numbers, for pencil-on-back or graphite transfer. `template`: 2 pt lines, 9-24 pt bold numbers, for cutting or painting up to the edge. Also changes the printed instructions |
| `--brand` | none | brand name for labels only |
| `--palette-json` | none | output of `brand_match.py`: `{"1": {"name", "code", "hex", "delta_e"}}` |
| `--scale` | 1.0 | print scale. 1.0 = sheets are true wall size (tracing templates). Below 1 gives a smaller reference set, flagged in red on the cover and on every sheet |
| `--preview-only` | off | skip the sheets |
| `--out` | required | output directory |

### paint_math.py flags

`--areas` (required), `--coats` and `--coverage-sqft-per-gal` (default to the values in
`paint_areas.json`), `--waste` (default 0.10), `--prices` (default `sample=6.98,quart=18.48,gallon=34.98`,
rough US big-box prices; pass current ones), `--out-prefix` (default `shopping_list` next to the areas
file), `--merge-pdf` + `--merged-out`.

Gallons per color = sq ft × coats / coverage × (1 + waste). The purchase is the cheapest mix of cans
that covers it, except that among mixes within 10% of the cheapest it takes the one with the least
paint (0.3 gal → 2 quarts, not a gallon; 0.8 gal → 1 gallon). Sample jars (8 oz = 1/16 gal) are used
only up to 2 per color and never mixed with cans. Leave `sample` out of `--prices` for exterior or
high-traffic walls, or for brands whose samples are labeled "for color testing only".

### Brand palettes

`brand_match.py --palette` takes `{"brand": "...", "colors": [{"name", "code", "hex"}]}`.
`example_brand_palette.json` is **made-up example data**. For a real job, the bot builds this file from
the chosen brand's published color list (web search for the brand's official color list or download,
recording name, code and hex for each color). Matches are always approximate: screen colors differ from
paint, so organizers should check physical chips. Brand terms often restrict reuse of their color data,
so build the palette per job and don't redistribute it.

## How it works

1. **Fit.** The picture is center-cropped to the wall's aspect ratio (no distortion and no padding; the
   cover states how much was cropped), then resampled to 24 px per wall inch. Walls over 12 MP of work
   area use a lower resolution.
2. **Colors.** A median filter removes noise but keeps edges. Then k-means in CIELAB (fixed seed) runs on a
   color histogram weighted by √count, so a small distinct feature like a sun keeps its own color.
3. **Cleanup.** A majority (mode) filter about `min-region-in`/4 wide comes first. Then tiny or thin regions
   merge into their most-touched neighbor, repeating until none are left. Colors are numbered lightest (1)
   to darkest, so painting in order lets dark colors cover mistakes.
4. **Outlines.** The boundaries between pixels become a graph, chained into polylines that meet exactly at
   junctions, smoothed with a Gaussian (ends pinned) and drawn as vectors. Sheets stay sharp at any zoom
   and the PDF stays small.
5. **Tiling.** Letter pages have 0.25 in margins, a header (sheet ID, "row r of R, col c of C", the wall
   span it covers) and a footer (colors on the sheet, a 2 in scale bar, a you-are-here grid). The tool
   picks portrait or landscape, whichever needs fewer sheets. Neighbors share shaded overlap bands with
   magenta crosshairs at band centers (corners and edge midpoints). Those land on the same wall point on
   every neighbor, so they stack exactly.
6. **Numbers.** A distance transform puts each number at the most interior point of every piece of a
   region, sized to fit, within that sheet's own map cell (the tile minus its right/bottom bands). The
   next sheet, taped over those bands, carries them, so no number is hidden or doubled. `paint_areas.json`
   reports `unlabeled_regions`, which should be 0.

## Checking a kit

```bash
pdfinfo kit.pdf                                             # every page 612 x 792 pt (letter), any orientation
pdftoppm -r 100 -png -f 1 -l 3 kit.pdf preview              # look at cover, map, shopping list
python3 sample/verify_kit.py kit.pdf 3 job/paint_areas.json # 3 = pages before the first sheet in the merged kit
```

`verify_kit.py` checks that every sheet's header ID/row/col matches its page position and appears on the
map. It also renders neighbor pairs at 200 dpi and checks that every inked pixel in one sheet's overlap band
is on the neighbor too.

## Sample

`sample/run_sample.sh` generates a synthetic sunrise image (`make_sample_image.py`, no licensing issues)
and builds:

- `sample/mural_sample.pdf`: 10×8 ft wall, 8 colors, 2 coats, trace, true size (176 sheets, ~0.6 MB)
- `sample/mural_sample_small.pdf`: the same design at 25% scale (12 sheets), easy to share
- `sample/shopping_list.{md,csv,pdf}` and rendered previews in `sample/previews/`

## Known limits

- **Sheet count.** A true-size kit needs about 2.2 sheets per sq ft of wall (176 for 10×8 ft), because each page
  loses margins, header, footer and overlap. An edge row or column can cover only a few inches of wall.
  Use `--scale` for reference sets; true size is for tracing.
- **Gradients become bands.** Smooth skies eat colors. If a small feature disappears, raise `--colors`
  (or simplify the picture first). Photos with lots of texture give many regions; raise `--min-region-in`.
- **Colors are screen colors.** Brand matches are nearest by CIEDE2000 and approximate. Very dark or
  very saturated colors may have no close paint (check `delta_e`; over ~5 is clearly off).
- **Paint math is an estimate.** Wall texture, porous or dark base coats, and brush loss change usage.
  Waste defaults to 10%, and default prices are rough. Primer isn't included.
- **Center-crop only.** Crop or pad the picture yourself first if the subject is off-center.
- Letter paper only; the overlap is in paper inches; very large walls lower the work resolution.
