# Week 1

Two of us, about 20 hours each. The goal is to kill the two risks that could
sink the whole thing, and to get real numbers on the board before 26 August.

## Uriel: the image work

1. `make data`, then read the layout the script prints. How many images per
   patient, are there repeat captures of the same field, is there a z-stack.
   That last question decides whether `refocus()` is real or simulated, so
   answer it first and write the answer in `docs/data-notes.md`.
2. Run `assess_field()` over a few hundred real fields. Plot the four numbers.
   The thresholds in `quality.py` are guesses; replace them with values that
   actually separate usable fields from unusable ones on this data.
3. Write `segmentation.py`: adaptiveThreshold, morphology, distanceTransform,
   watershed, findContours, then the shape filters. Count white cells first,
   parasites second. White cells are bigger and easier and they are the
   denominator, so getting them right matters more than it looks.

## Alex: AWS and the container

1. Open the AWS account, turn on a budget alarm before anything else runs.
2. Dockerfile for `linux/arm64` with OpenCV 5. Get `make test` and
   `make bench` green inside it under emulation.
3. Subscribe to the COOL AMI on Marketplace but **do not start the 7 day trial
   yet**. We want that week to land on the benchmark, in week 8.
4. Stand up the smallest thing that works end to end: S3 upload, EventBridge,
   one Batch job on `c8g` Spot that runs `assess_field()` over one tile and
   writes the numbers to DynamoDB. Nothing clever. Just prove the path.

## Both, end of week

Run the benchmark on `c8g` and on `c7i` and commit the JSON. Even without COOL
those two files are the control for everything we claim later.

## Already done

- `density.py` and its tests: the stopping rule is written and covered. This is
  the agentic claim, and it works today.
- `quality.py`: the four numbers the agent reads.
- `virtual_stage.py`: `move_stage()` and `refocus()`.
- `bench/cool_bench.py`: ten OpenCV calls timed with medians and IQR.

## What the local x86 baseline already tells us

From `bench/results/local-x86.json`, on a 1024px tile:

```
findContours       73.5 ms   <-- dominates everything else
adaptiveThreshold  10.3 ms
medianBlur          6.6 ms
GaussianBlur        3.9 ms
warpPerspective     2.5 ms
Laplacian           2.1 ms
```

`findContours` is 70% of the per-tile cost, and it is in the set COOL
accelerates. Optimise the segmentation loop around that, and make sure the COOL
comparison reports it separately rather than burying it in a total.
