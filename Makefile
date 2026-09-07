.PHONY: setup setup-train setup-detect data data-tanzania test bench train-mlp train-cnn export-detect train-yolo train-frcnn clean

setup:                     ## create the venv and install pinned deps
	uv venv --python 3.12
	uv pip install -r requirements.txt -e .

setup-train:               ## separate venv for training the patch network (torch needs numpy 1.x)
	uv venv --python 3.12 .venv-train
	uv pip install --python .venv-train/bin/python -r requirements-train.txt

setup-detect:              ## separate venv for the box detectors (torch + ultralytics, ~3 GB)
	uv venv --python 3.12 .venv-detect
	uv pip install --python .venv-detect/bin/python -r requirements-detect.txt

data:                      ## download the NLM thick smear set (3.9 GB, resumable)
	.venv/bin/python data/download.py

data-tanzania:             ## download the held-out Tanzania set (3.0 GB, CC0)
	.venv/bin/python data/download_tanzania.py

test:                      ## run the unit tests
	.venv/bin/pytest -q

train-mlp:                 ## parasite feature classifier: candidates -> numpy MLP -> ONNX
	cd eval && ../.venv/bin/python parasite_candidates.py --images 100
	cd eval && ../.venv/bin/python parasite_train.py

train-cnn:                 ## parasite patch classifier: patches -> torch CNN -> ONNX -> cv.dnn check
	cd eval && ../.venv/bin/python parasite_patches.py --images 100
	cd eval && ../.venv-train/bin/python parasite_cnn_train.py
	cd eval && ../.venv/bin/python parasite_cnn_check.py

export-detect:             ## tile the FIT patients into a YOLO + COCO detection set
	.venv/bin/python eval/export_detection.py --images-per-patient 3

train-yolo:                ## YOLO26 on the tiles, tracked in ClearML (needs a GPU)
	.venv-detect/bin/python train/yolo26.py

train-frcnn:               ## Faster R-CNN on the same tiles (needs a GPU)
	.venv-detect/bin/python train/frcnn.py

bench:                     ## run the COOL benchmark on this machine
	.venv/bin/python bench/cool_bench.py --runs 100

clean:
	rm -rf .venv .venv-train .venv-detect .pytest_cache **/__pycache__
