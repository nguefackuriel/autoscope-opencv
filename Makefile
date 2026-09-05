.PHONY: setup setup-train data data-tanzania data-uninfected data-vivax test bench docker docker-arm64 train-mlp train-cnn chain clean

setup:                     ## create the venv and install pinned deps
	uv venv --python 3.12
	uv pip install -r requirements.txt -e .

setup-train:               ## separate venv for training the patch network (torch needs numpy 1.x)
	uv venv --python 3.12 .venv-train
	uv pip install --python .venv-train/bin/python -r requirements-train.txt

data:                      ## download the NLM thick smear set (3.9 GB, resumable)
	.venv/bin/python data/download.py

data-tanzania:             ## download the held-out Tanzania set (3.0 GB, CC0)
	.venv/bin/python data/download_tanzania.py

data-uninfected:           ## NLM uninfected thick smears (2.5 GB): specificity, frozen test set
	.venv/bin/python data/download_nlm_extra.py --set uninfected

data-vivax:                ## NLM P. vivax thick smears (5.0 GB): other-species test, frozen
	.venv/bin/python data/download_nlm_extra.py --set vivax

test:                      ## run the unit tests
	.venv/bin/pytest -q

train-mlp:                 ## parasite feature classifier: candidates -> numpy MLP -> ONNX
	cd eval && ../.venv/bin/python parasite_candidates.py --images 100
	cd eval && ../.venv/bin/python parasite_train.py

train-cnn:                 ## parasite patch classifier: mined patches -> torch CNN (48 px) -> ONNX -> cv.dnn check
	cd eval && ../.venv/bin/python parasite_patches.py --mine --per-patient 3 --patch 56 --hard-min 0.2 --easy-keep 0.1 --uninfected-images -1 --miner-model ../models/parasite_cnn_chain2_32px.onnx --miner-size 32 --out parasite_patches_mined_56.npz
	cd eval && ../.venv-train/bin/python parasite_cnn_train.py --patches parasite_patches_mined_56.npz --crop 48 --epochs 20 --val-file calibration_patients.json
	cd eval && ../.venv/bin/python parasite_cnn_check.py

chain:                     ## train-cnn, then every held-out and specificity number, about seven hours: caffeinate -dims make chain
	bash eval/chain.sh

bench:                     ## run the COOL benchmark on this machine
	.venv/bin/python bench/cool_bench.py --runs 100

docker:                    ## build the stock-OpenCV image for this machine and run the tests in it
	docker build -t autoscope .
	docker run --rm autoscope

docker-arm64:              ## same image for linux/arm64 under emulation: correctness only, never timings
	docker build --platform linux/arm64 -t autoscope:arm64 .
	docker run --rm --platform linux/arm64 autoscope:arm64

clean:
	rm -rf .venv .venv-train .pytest_cache **/__pycache__
