import pandas as pd

from src.data.llvip import HEIGHT, WIDTH
from src.data.split import TEST_SCENES, META
from tests.conftest import needs_data

pytestmark = needs_data
SPLITS = ("train", "val", "test")


def test_every_pair_in_exactly_one_split(cocos):
    stems = [i["stem"] for s in SPLITS for i in cocos[s]["images"]]
    assert len(stems) == len(set(stems)) == 15488


def test_test_split_is_official_llvip_test(cocos):
    assert {i["scene"] for i in cocos["test"]["images"]} == TEST_SCENES
    assert all(i["file_name"].startswith("test/") for i in cocos["test"]["images"])
    assert all(i["file_name"].startswith("train/") for s in ("train", "val") for i in cocos[s]["images"])


def test_no_location_in_two_splits(cocos):
    locs = {s: {i["location"] for i in cocos[s]["images"]} for s in SPLITS}
    assert not locs["train"] & locs["val"]
    assert not locs["train"] & locs["test"]
    assert not locs["val"] & locs["test"]


def test_no_cross_split_near_duplicates():
    assert pd.read_csv(META / "cross_split_near_dups.csv").empty


def test_annotations_valid(cocos):
    for s in SPLITS:
        ids = {i["id"] for i in cocos[s]["images"]}
        ann_ids = [a["id"] for a in cocos[s]["annotations"]]
        assert len(ann_ids) == len(set(ann_ids))
        for a in cocos[s]["annotations"]:
            x, y, w, h = a["bbox"]
            assert a["image_id"] in ids and a["category_id"] == 1
            assert w > 0 and h > 0 and x >= 0 and y >= 0
            assert x + w <= WIDTH + 1 and y + h <= HEIGHT + 1


def test_all_boxes_present(cocos):
    assert sum(len(cocos[s]["annotations"]) for s in SPLITS) == 42437 - 5  # minus degenerate boxes dropped in src.data.split
