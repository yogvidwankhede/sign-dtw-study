# One-command reproduction. `make all` needs ~1 TB of network transfer for the full 250-sign vocabulary and many hours;
# use `make quick` for a 10-sign run. Landmarks are cached in data/landmarks, so reruns skip finished signs.
PY = .venv/bin/python
VOCAB ?= 10

setup:
	uv venv --python 3.11 .venv && uv pip install --python $(PY) -r requirements.txt && uv pip install --python $(PY) --no-deps -r requirements-extra.txt

test:
	$(PY) -m pytest -q tests

extract:
	$(PY) -u -m signstudy.extract $(VOCAB) 8

tune:
	$(PY) -u -m signstudy.pipeline tune --split val --vocab $(VOCAB)

main:
	$(PY) -u -m signstudy.pipeline main --split test --vocab $(VOCAB)

ablate:
	$(PY) -u -m signstudy.pipeline ablate --split test --vocab $(VOCAB)

timing:
	NUMBA_NUM_THREADS=1 $(PY) -m signstudy.timing --threads 1 --vocab $(VOCAB)
	$(PY) -m signstudy.timing --threads 8 --vocab $(VOCAB)

report:
	$(PY) -m signstudy.report

paper:
	cd paper && tectonic paper.tex

quick: extract tune main ablate timing report

all: test
	$(MAKE) quick VOCAB=250
