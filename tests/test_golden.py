import json

from src.data.golden import MANIFEST, sha256
from src.data.llvip import Pair
from tests.conftest import needs_data

pytestmark = needs_data


def test_golden_files_unchanged():
    manifest = json.loads(MANIFEST.read_text())
    changed = []
    for stem, hashes in manifest["sha256"].items():
        pair = Pair(stem, "test")
        actual = {"visible": sha256(pair.visible()), "infrared": sha256(pair.infrared()),
                  "annotation": sha256(pair.annotation())}
        if actual != hashes:
            changed.append(stem)
    assert not changed, f"golden files changed: {changed[:10]}"


def test_golden_is_subset_of_test(cocos):
    manifest = json.loads(MANIFEST.read_text())
    test_stems = {i["stem"] for i in cocos["test"]["images"]}
    assert set(manifest["stems"]) <= test_stems
    assert set(manifest["ci_stems"]) <= set(manifest["stems"])
    assert {i["stem"] for i in cocos["golden"]["images"]} == set(manifest["stems"])
    assert {i["stem"] for i in cocos["golden_ci"]["images"]} == set(manifest["ci_stems"])


def test_golden_covers_all_conditions_and_locations(cocos):
    imgs = cocos["golden"]["images"]
    assert {i["condition"] for i in imgs} == {"dark", "dim", "lit"}
    assert {i["location"] for i in imgs} == {i["location"] for i in cocos["test"]["images"]}
