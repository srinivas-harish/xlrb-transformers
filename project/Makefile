.PHONY: dev install venv

venv:
	python3 -m venv .venv

install: venv
	. .venv/bin/activate && pip install --upgrade pip && \
	pip install fastapi uvicorn celery[redis] redis requests tqdm datasets scikit-learn transformers==4.44.2 && \
	pip install "torch==2.7.1+cu128" "torchvision==0.22.1+cu128" "torchaudio==2.7.1+cu128" --index-url https://download.pytorch.org/whl/cu128 && \
	pip install honcho

dev:
	. .venv/bin/activate && honcho -f Procfile.dev start
