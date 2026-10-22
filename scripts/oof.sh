#!/usr/bin/env bash
# Out-of-fold predictions on the train split for label-error mining (Phase 6): train on one half of
# the train locations, predict the other half, then swap. Runs locally.
set -e
FOLD_A="1,2,3,4,8,16,17,18"; FOLD_B="5,6,7,9,10,12,13,25"
MOD=${MOD:-thermal}; EPOCHS=${EPOCHS:-16}
for pair in "A:$FOLD_A:$FOLD_B" "B:$FOLD_B:$FOLD_A"; do
  IFS=: read name train_locs pred_locs <<< "$pair"
  out=runs/oof/$MOD-fold$name
  uv run python -m src.train --modality $MOD --locations $train_locs --epochs $EPOCHS --batch-size 8 \
      --effective-batch 16 --workers 6 --out $out --name oof-$MOD-fold$name 2>&1 | grep --line-buffered -E '^\{"epoch|Error|Traceback'
  uv run python - "$pred_locs" "$out/heldout.json" <<'PY'
import json, sys
locs = {int(x) for x in sys.argv[1].split(",")}
c = json.load(open("data/annotations/train.json"))
ids = {i["id"] for i in c["images"] if i["location"] in locs}
json.dump({**c, "images": [i for i in c["images"] if i["id"] in ids],
           "annotations": [a for a in c["annotations"] if a["image_id"] in ids]}, open(sys.argv[2], "w"))
PY
  uv run python -m src.predict --gt $out/heldout.json --modality $MOD --checkpoint $out/best \
      --out $out/pred_heldout.json --workers 6 2>&1 | tail -1
done
echo OOF_DONE
