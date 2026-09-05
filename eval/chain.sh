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
LOG=results/chain3_2026-09-05.log
MINER=../models/parasite_cnn_chain2_32px.onnx
step() { echo; echo "===== $(date +%H:%M) $*"; }

{
step "patches: every fit image, hard negatives >= 0.2 under the chain-2 network, 10% of the easy ones, 56 px cut; all 569 clean fit-half fields; calibration patients 3 images each, every candidate"
$PY parasite_patches.py --mine --per-patient 3 --patch 56 --hard-min 0.2 --easy-keep 0.1 \
    --uninfected-images -1 --miner-model "$MINER" --miner-size 32 --out parasite_patches_mined_56.npz

step "train the patch network: 48 px, 20 epochs, validation = calibration_patients.json"
$PYT parasite_cnn_train.py --patches parasite_patches_mined_56.npz --crop 48 --epochs 20 --val-file calibration_patients.json

step "cv.dnn agrees with torch?"
$PY parasite_cnn_check.py

step "rescore candidates: calibration, then held-out"
$PY parasite_scores.py --group calibration
$PY parasite_scores.py --group holdout

step "choose the threshold on the calibration patients"
$PY choose_threshold.py
T=$($PY -c 'import json; print(json.load(open("results/threshold_fixedstats.json"))["threshold"])')

step "parasite_calibrate at $T"
$PY parasite_calibrate.py --fixed "$T"

step "parasite_eval held-out at $T"
$PY parasite_eval.py --holdout --threshold "$T" --tag holdout_cnn_mined

step "agent_fields at $T (609 fields)"
$PY agent_fields.py --threshold "$T"

step "agent_eval"
$PY agent_eval.py

step "negative_fields at $T (1141 fields, all 50 patients; the eval reports the frozen half)"
$PY negative_fields.py --threshold "$T"

step "negative_eval, frozen half"
$PY negative_eval.py

step "density bands and clean-field false alarms by half"
$PY density_bands.py

step "done"
} 2>&1 | tee -a "$LOG"
