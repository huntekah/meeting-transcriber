.PHONY: help install run dev test clean lint format

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies using uv"
	@echo "  make run        - Start the ASR service"
	@echo "  make dev        - Start with auto-reload (development mode)"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run code linters"
	@echo "  make format     - Format code"
	@echo "  make clean      - Clean temporary files"

install:
	uv sync

run:
	cd src && uvicorn asr_service.main:app --host 0.0.0.0 --port 8000

dev:
	cd src && uvicorn asr_service.main:app --host 0.0.0.0 --port 8000 --reload

test:
	uv run pytest

lint:
	uv run ruff check src/
	uv run mypy src/

format:
	uv run ruff format src/
	uv run ruff check --fix src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
