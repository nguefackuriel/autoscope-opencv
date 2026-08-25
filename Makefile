.PHONY: setup data data-tanzania test bench clean

setup:                     ## create the venv and install pinned deps
	uv venv --python 3.12
	uv pip install -r requirements.txt -e .

data:                      ## download the NLM thick smear set (3.9 GB, resumable)
	.venv/bin/python data/download.py

data-tanzania:             ## download the held-out Tanzania set (3.0 GB, CC0)
	.venv/bin/python data/download_tanzania.py

test:                      ## run the unit tests
	.venv/bin/pytest -q

bench:                     ## run the COOL benchmark on this machine
	.venv/bin/python bench/cool_bench.py --runs 100

clean:
	rm -rf .venv .pytest_cache **/__pycache__
