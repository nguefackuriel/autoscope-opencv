#!/bin/bash
# Chain 3, 5 September 2026: sparse-field false alarms without new data.
# docs/plan-2026-09-05-hospital-grade.md, step 1. Run it as
#
#     caffeinate -dims bash eval/chain.sh
#
# and leave the lid open: caffeinate does not stop lid or scheduled sleep
# (chains 1 and 2 both slept overnight). Every step appends to the log with
# a timestamp so a reader can see what ran and how long it took. The whole
# chain is about seven hours on the laptop, five of them training.
#
# The miner is the chain-2 network at 32 px, kept as models/parasite_cnn_chain2_32px.onnx:
# it scores every candidate so the hard negatives can be kept, and it is the
# control the report compares against.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
PYT=../.venv-train/bin/python
LOG=results/chain_$(date +%Y-%m-%d)_${MODEL:-cnn}.log
CNN_MINER=../models/parasite_cnn_chain2_32px.onnx   # chain 3's miner, 32 px; MINER (env) is the heat-cnn path's
step() { echo; echo "===== $(date +%H:%M) $*"; }
# SKIP_TRAIN=1 bash eval/chain.sh re-measures everything with the network as
# it is: for a change to the white cells, the candidates or the rule.
# MODEL=heatmap bash eval/chain.sh trains and measures the learned detector
# (parasite_tiles.py, parasite_heatmap_train.py) instead of the patch network.
# MODEL=heat-cnn trains the patch network on the heat map's candidates and
# measures the pair.
SKIP_TRAIN=${SKIP_TRAIN:-0}
MODEL=${MODEL:-cnn}

{
if [ "$SKIP_TRAIN" = "0" ] && [ "$MODEL" = "heatmap" ]; then
step "tiles: every fit image at half resolution, 6 tiles each, 2 per clean fit-half field"
$PY parasite_tiles.py --per-image 6 --per-clean 2 --uninfected-images -1 --out parasite_tiles.npz

step "train the heat-map detector: U-Net 12-24-48-96, 16 epochs of 4 000 tiles, validation = calibration_patients.json"
$PYT parasite_heatmap_train.py --tiles parasite_tiles.npz --epochs 16 --tiles-per-epoch 4000 --val-file calibration_patients.json

step "cv.dnn agrees with torch, and runs a whole field?"
$PY parasite_heatmap_check.py
elif [ "$SKIP_TRAIN" = "0" ] && [ "$MODEL" = "heat-cnn" ]; then
# Chain 4 mined the heat map's candidates alone: 72 000 patches, two thirds
# positive, 2 600 from clean fields, and the judge came out strict on
# parasites and blind to clean-field look-alikes (results-parasites.md 16c).
# Chain 5 proposes with both, keeps every clean-field candidate.
# Chain 6 (11 September): mine with the judge being improved (MINER), because a
# miner that already rejects a look-alike hides it from the next judge
# (results-parasites.md 16e); CLEAN_KEEP thins the clean-field candidates.
HEAT_MINER=${MINER:-../models/parasite_cnn.onnx}
CLEAN_KEEP=${CLEAN_KEEP:-1.0}
step "patches: every fit image, candidates from the top-hat AND the heat map, hard negatives >= 0.2 under $HEAT_MINER, 10% of the easy ones, clean-field candidates kept at $CLEAN_KEEP, 56 px cut; calibration patients 3 images each"
$PY parasite_patches.py --mine --candidates both --per-patient 3 --patch 56 --hard-min 0.2 --easy-keep 0.1 --easy-keep-clean "$CLEAN_KEEP" \
    --uninfected-images -1 --miner-model "$HEAT_MINER" --miner-size 48 --out parasite_patches_heat_56.npz

step "train the judge for the heat map's candidates: 48 px, 20 epochs, validation = calibration_patients.json -> models/parasite_cnn_heat.onnx"
$PYT parasite_cnn_train.py --patches parasite_patches_heat_56.npz --crop 48 --epochs 20 --val-file calibration_patients.json \
    --model-out ../models/parasite_cnn_heat.onnx --out-tag heat

step "cv.dnn agrees with torch?"
$PY parasite_cnn_check.py --tag heat --model ../models/parasite_cnn_heat.onnx
elif [ "$SKIP_TRAIN" = "0" ]; then
step "patches: every fit image, hard negatives >= 0.2 under the chain-2 network, 10% of the easy ones, 56 px cut; all 569 clean fit-half fields; calibration patients 3 images each, every candidate"
$PY parasite_patches.py --mine --per-patient 3 --patch 56 --hard-min 0.2 --easy-keep 0.1 \
    --uninfected-images -1 --miner-model "$CNN_MINER" --miner-size 32 --out parasite_patches_mined_56.npz

step "train the patch network: 48 px, 20 epochs, validation = calibration_patients.json"
$PYT parasite_cnn_train.py --patches parasite_patches_mined_56.npz --crop 48 --epochs 20 --val-file calibration_patients.json

step "cv.dnn agrees with torch?"
$PY parasite_cnn_check.py
else
step "measure only: network unchanged ($(git -C .. log --oneline -1))"
fi

step "rescore candidates: calibration, then held-out ($MODEL)"
$PY parasite_scores.py --model "$MODEL" --group calibration
$PY parasite_scores.py --model "$MODEL" --group holdout

step "choose the threshold on the calibration patients"
$PY choose_threshold.py
T=$($PY -c 'import json; print(json.load(open("results/threshold_fixedstats.json"))["threshold"])')

step "parasite_calibrate at $T"
$PY parasite_calibrate.py --fixed "$T"

step "parasite_eval held-out at $T"
$PY parasite_eval.py --holdout --threshold "$T" --model "$MODEL" --tag "holdout_$MODEL"

step "agent_fields at $T (609 fields)"
$PY agent_fields.py --threshold "$T" --model "$MODEL"

step "agent_eval"
$PY agent_eval.py

step "negative_fields at $T (1141 fields, all 50 patients; the eval reports the frozen half)"
$PY negative_fields.py --threshold "$T" --model "$MODEL"

step "negative_eval, frozen half"
$PY negative_eval.py

step "density bands and clean-field false alarms by half"
$PY density_bands.py

step "decision rule: design sweep on the fit half, dilution"
$PY negative_eval.py --design --sweep | tail -45
$PY dilution_eval.py

step "done"
} 2>&1 | tee -a "$LOG"
