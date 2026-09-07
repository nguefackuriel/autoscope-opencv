# Box detectors for parasites: method, and the traps

Branch `b/parasite-detectors`. **No numbers yet** — nothing has been trained.
This records the method and the constraints before any result exists, so that
when a number arrives it can be read against what we said we would do rather
than against what we wish we had done.

The target to beat is the patch network on the same held-out patients:
**F1 0.762, precision 0.841** (`docs/results-parasites.md`).

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
comparable with 0.762. `eval/detector_eval.py` refuses to sweep a threshold on
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
