import json

import pytest

from src.data.llvip import LLVIP, ROOT

ANNOTATIONS = ROOT / "data" / "annotations"

needs_data = pytest.mark.skipif(not (LLVIP / "Annotations").exists() or not ANNOTATIONS.exists(),
                                reason="LLVIP / built annotations not present (run `dvc pull` + `dvc repro`)")


@pytest.fixture(scope="session")
def cocos():
    return {s: json.loads((ANNOTATIONS / f"{s}.json").read_text())
            for s in ("train", "val", "test", "golden", "golden_ci")}
