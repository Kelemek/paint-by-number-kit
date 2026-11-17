# Paint by Number Kit scripts

Helper scripts for the Paint by Number Grok Bot template. They turn a picture into a printable paint-by-number kit: numbered 8.5x11 sheets, a placement map, and a paint shopping list, for anything from a small canvas to a full wall.

## Setup

```bash
git clone https://github.com/Kelemek/paint-by-number-kit ~/mural-kit
~/mural-kit/get_model.sh
python3 -m pip install --user --break-system-packages -r ~/mural-kit/requirements.txt
python3 ~/mural-kit/mural_tiler.py --help
```

Dependency versions in `requirements.txt` are pinned on purpose for safety and are bumped deliberately.

## License

This project is licensed under the [MIT License](LICENSE).

## Files

- `mural_tiler.py` reduces the picture to N colors, numbers the regions, tiles it into sheets, and builds the kit PDF. Face and eye priority are on by default.
- `brand_match.py` matches each kit color to the nearest color in a brand palette JSON (always approximate; check paint chips).
- `paint_math.py` turns each color's area into a shopping list with can sizes and merges it into the kit.
- `example_brand_palette.json` shows the palette format.
- `get_model.sh` downloads the YuNet face detector (OpenCV Zoo, MIT license) into `models/`.
- `TILER_REFERENCE.md` lists every tiler flag and the known limits.

Paint amounts and color matches are estimates only. This project isn't affiliated with, endorsed by, or sponsored by xAI, SpaceXSI, or Grok.
