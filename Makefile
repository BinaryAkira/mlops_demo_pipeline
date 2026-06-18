# ============================
# Makefile for MLOps pipeline
# ============================

.PHONY: train test lint

train:
    python main.py

test:
    pytest -q

lint:
    flake8 src tests
