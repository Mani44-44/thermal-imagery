"""LLVIP dataset layout helpers: pairs, scenes and VOC annotations."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLVIP = ROOT / "data" / "raw" / "LLVIP"
WIDTH, HEIGHT = 1280, 1024


@dataclass(frozen=True)
class Pair:
    stem: str          # e.g. "010001"
    official_split: str  # "train" or "test"

    @property
    def scene(self) -> int:
        # The two-digit filename prefix is the scene ID (26 scenes, see docs/data-checks.md).
        return int(self.stem[:2])

    def visible(self, root: Path = LLVIP) -> Path:
        return root / "visible" / self.official_split / f"{self.stem}.jpg"

    def infrared(self, root: Path = LLVIP) -> Path:
        return root / "infrared" / self.official_split / f"{self.stem}.jpg"

    def annotation(self, root: Path = LLVIP) -> Path:
        return root / "Annotations" / f"{self.stem}.xml"


def list_pairs(root: Path = LLVIP) -> list[Pair]:
    pairs = [Pair(p.stem, split)
             for split in ("train", "test")
             for p in sorted((root / "visible" / split).glob("*.jpg"))]
    assert pairs, f"no images under {root}"
    return pairs


def read_boxes(xml_path: Path) -> list[dict]:
    """Return boxes as dicts with xyxy pixel coords (VOC convention) and the truncated flag."""
    boxes = []
    for obj in ET.parse(xml_path).getroot().iter("object"):
        assert obj.findtext("name") == "person", f"unexpected class in {xml_path}"
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append({"xyxy": [x1, y1, x2, y2], "truncated": int(obj.findtext("truncated") or 0)})
    return boxes
