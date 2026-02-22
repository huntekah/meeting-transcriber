.PHONY: help install run dev run-cli test test-coverage test-backend test-frontend clean lint format

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies using uv"
	@echo "  make run           - Start the ASR service"
	@echo "  make dev           - Start with auto-reload (development mode)"
	@echo "  make run-cli       - Start the CLI frontend"
	@echo "  make test          - Run all tests"
	@echo "  make test-coverage - Run tests with coverage report"
	@echo "  make test-backend  - Run backend tests only"
	@echo "  make test-frontend - Run frontend tests only"
	@echo "  make lint          - Run code linters"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean temporary files"

install:
	uv sync

run:
	cd src && uv run uvicorn asr_service.main:app --host 0.0.0.0 --port 8000

dev:
	cd src && uv run uvicorn asr_service.main:app --host 0.0.0.0 --port 8000 --reload

run-cli:
	cd src && uv run --group cli python -m cli_frontend

test:
	uv run --all-groups pytest

test-coverage:
	uv run --all-groups pytest --cov=src/asr_service --cov=src/cli_frontend --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

test-backend:
	uv run pytest tests/backend/ -v

test-frontend:
	uv run --group cli pytest tests/frontend/ -v

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
