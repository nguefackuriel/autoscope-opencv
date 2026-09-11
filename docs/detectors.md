# Box detectors for parasites: method, and the traps

Branch `b/parasite-detectors`. **No numbers yet** — nothing has been trained.
This records the method and the constraints before any result exists, so that
when a number arrives it can be read against what we said we would do rather
than against what we wish we had done.

The target to beat is the patch network on the same held-out patients:
**F1 0.763, precision 0.790, recall 0.739, count bias -6.4%**
(`docs/results-parasites.md`, patch CNN at the corrected 0.75 threshold).

Quote the count bias alongside the F1, always. The threshold moved from 0.80
to 0.75 not because F1 improved -- it is 0.762 against 0.763, the same curve --
but because 0.80 ran the count 14.4% low, and the count is the denominator of
every density a clinician reads.

## The split is inherited, not chosen

`eval/results/wbc_sweep.json` → `fit_patients` is 100 patient ids. The held-out
50 are the complement and are not listed anywhere. Everything published by this
project uses that one artifact, so the detectors read it too, via
`eval/splits.py`.

Training needs a validation set for checkpoint and threshold selection, and it
must not be the held-out 50 — choosing a model by its score there is fitting on
it. So `splits.train_val()` carves validation out of the **fit** patients,
80/20, split by patient rather than by image: two fields from one slide share a
stain, a focus and a camera, so splitting by image would leak.

The held-out patients are never exported, never tiled, never seen until a model
and a threshold are already fixed.

## The dataset has no boxes, so we measure them

`White_Blood_Cell` is a single click with no size. `Parasite` is a centre plus a
rim point, and `annotations.py` is explicit that the annotator radius (median
22 px) is "an upper bound on the object, not its size" — the chromatin dot is
about 10 px. A box drawn from that radius is roughly four times the area of the
thing it should contain.

So `src/autoscope/boxes.py` takes the box from the pixels instead: threshold
`parasites.tophat()` at half its local peak near the annotated centre, and bound
the component. The search allows ±12 px for the click, because the annotated
centre is a median 7.9 px from the most purple pixel.

An annotation whose stain shows no peak yields **no box at all**. That is a real
answer, not a gap to paper over, and the export reports the drop rate so it can
be argued with.

### The trap this creates

These boxes are an algorithm's opinion. Scoring a detector by IoU against them
would mostly measure how well it imitates `boxes.py`, and a perfect mAP would
mean we had reproduced the top-hat generator we set out to beat.

So detections are scored the way everything else here is scored: **box centre
against annotated centre, greedy nearest, 15 px tolerance**, via the same
`wbc_eval.match` that `parasite_eval.py` uses. That is what makes a number
comparable with 0.763. `eval/detector_eval.py` refuses to sweep a threshold on
the held-out split.

## Why tiles

A parasite is ~10 px in a ~3000 px field. Resizing that field to 640 px makes it
2 px, below what any box head can regress. So tiles are cut at **native
resolution**, 640 px with 128 px of overlap, and inference tiles the same way
and merges back. A box that straddles a tile edge is dropped, which is safe only
because the overlap guarantees a neighbouring tile holds it whole —
`tests/test_export_detection.py` pins exactly that.

Two model-specific consequences:

- **Faster R-CNN**: torchvision's default FPN anchors are (32…512) px, chosen
  for COCO. Against a 10 px object every anchor has near-zero IoU, the RPN
  labels them all background, and training converges to a detector that finds
  nothing while the loss curve looks healthy. `train/frcnn_model.py` uses
  (4, 8, 16, 32, 64).
- **YOLO26**: `mosaic` stitches four images and resizes, roughly halving every
  object, and default `scale` is ±50%. Both are off or turned right down in
  `train/yolo26.py`. Colour jitter is kept small because the tiles are already
  stain-normalised — jittering hard would be undoing stage 2 on purpose.

Both models train on the **same jpg tiles**, YOLO from `data.yaml` and Faster
R-CNN from the COCO json, so any difference between them is the model and not
the data.

## What could make this fail, honestly

- **The drop rate could be high.** If `boxes.py` refuses a large share of
  annotations, the detectors train on a biased subset — the parasites with
  clean stain — and will look better in training than they are. The export
  prints this number; read it before reading anything else.
- **Tiling multiplies the dataset.** 36 tiles per field at 3 images per patient
  over 100 fit patients is on the order of 10⁴ tiles. That is fine for a GPU and
  hopeless on a laptop.
- **Point matching at 15 px is generous to a box detector** in one direction: a
  box roughly the right place scores a hit even if its size is wrong. That is
  intentional, since size has no ground truth, but it means these models are not
  being asked to do the one thing box detectors are usually judged on. Say so in
  the report.

## Order of operations

```
python eval/export_detection.py --images-per-patient 3
python train/yolo26.py            # or train/frcnn.py
python eval/detector_eval.py --backend yolo --weights <best> --split val --sweep
python eval/detector_eval.py --backend yolo --weights <best> --split holdout --threshold <chosen>
```

The last line runs **once**.

On Colab, `notebooks/colab_detectors.ipynb` walks the same order with the
caching and the split checks wired in. It uses Colab's preinstalled torch
rather than the pins in `requirements-detect.txt`, because forcing the pins
re-downloads ~3 GB and can break the CUDA build; it records the versions it
actually ran with in `eval/results/colab_env.json` so a result stays
attributable. On the AWS instance, use the pins.

One layout trap worth knowing: `data/download.py` extracts to
`data/raw/ThickBloodSmears_150/`, while the older eval scripts scan
`data/raw/` directly. `splits.patient_root()` resolves either and raises if it
finds neither, because the alternative is a run that reports a split of zero
patients and an F1 of 0.000 without ever saying why.

## 11 September: wired into the pipeline, and the target as it stands

Merged with `develop` (chains 3 to 5, the white cell gate, the decision
rule). Three things changed around the method above; the method itself did
not.

**The target moved.** On the same 50 held-out patients, 99 images, 2 per
patient (`docs/results-parasites.md` §15b): the shipped pair reads
precision 0.814, recall 0.783, **F1 0.798**, count bias −3.8%, at
threshold 0.53; the heat-map pairs of §16 reach F1 0.852 to 0.862 and are
not shipped, because a detector enters `develop` on four numbers at once:
density error per slide through the WHO protocol (25.7%), parasite bias
on the 1 000 to 10 000/µL band (+9.6%), false parasites per clean field
(0.06, floor 19/µL) and the limit of detection by dilution (500/µL). F1
is the first number, not the last.

**The pipeline runs a box detector like any other model.** The tiling,
the dedupe and the two backends moved from `detector_eval.py` into
`autoscope.boxes` (`BoxScorer`), and `parasites.find_parasites` /
`counter.count_field` take `model="yolo"` or `"frcnn"` with `--weights`
(or `AUTOSCOPE_YOLO_WEIGHTS`). So the whole re-measure is one line, in a
venv that has torch:

```
make setup-detect
SKIP_TRAIN=1 MODEL=yolo WEIGHTS=models/yolo26/best.pt PY=../.venv-detect/bin/python bash eval/chain.sh
```

It rescores the calibration and held-out patients, chooses the threshold
by the count-matching rule (`choose_threshold.py`, the count that equals
every annotated parasite on the 20 calibration patients, not best F1),
then the held-out detector table, the 609 held-out fields, the agent
replay, the 1 141 clean fields, the decision rule, the design sweep and
the dilution. Results land in the canonical files; keep them under a
`_yolo` suffix if the shipped pair stays shipped (see
`results-parasites.md` §16b for the convention).

**Clean fields are in the export.** `export_detection.py` now cuts
background tiles from the fit half of the uninfected set
(`--uninfected-fields`, every field by default, `--clean-tiles` per field).
Every negative tile before came from an infected image; on a clean field
the platelets are the whole population, and the patch network's false
alarms on clean fields fell by a factor of 22 the day they entered its
training set. The frozen half is never read.

**Validation set**: use the 20 calibration patients
(`eval/results/calibration_patients.json`) rather than a fresh 80/20 cut,
so the validation tables line up with the patch network's. `splits.train_val`
still exists; `--val-file calibration_patients.json` is the convention
elsewhere.

**Runtime.** These models run in torch and ultralytics. The COOL benchmark
and the "OpenCV 5" story need the count to run in `cv.dnn`; a YOLO ONNX
export can, a Faster R-CNN cannot easily. Settle this before spending GPU
hours on Faster R-CNN. On the laptop (Intel macOS) torch stops at 2.2, so
`requirements-detect.txt` installs only on Linux or Colab; the pipeline
side is tested with a fake backend (`tests/test_boxes_runtime.py`).
