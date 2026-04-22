from __future__ import annotations

from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CANDIDATES = (
    REPO_ROOT / "app_icon.png",
    REPO_ROOT / "src" / "helios_viewer" / "assets" / "app_icon.png",
)
OUTPUT_PNG = REPO_ROOT / "src" / "helios_viewer" / "assets" / "app_icon.png"
OUTPUT_ICO = REPO_ROOT / "src" / "helios_viewer" / "assets" / "app_icon.ico"


def find_source() -> Path:
    for candidate in SOURCE_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find the canonical app_icon.png asset.")


def main() -> int:
    source = find_source()
    image = Image.open(source).convert("RGBA")

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != OUTPUT_PNG.resolve():
        image.save(OUTPUT_PNG)
    image.save(OUTPUT_ICO, sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (24, 24), (16, 16)])
    print(f"Regenerated {OUTPUT_ICO.name} from canonical source {source.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
