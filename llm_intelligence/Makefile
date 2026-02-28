.PHONY: help install run dev test test-coverage lint format audit clean

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies using uv"
	@echo "  make run           - Start the LLM intelligence service (port 8001)"
	@echo "  make dev           - Start with auto-reload (development mode)"
	@echo "  make test          - Run all tests"
	@echo "  make test-coverage - Run tests with coverage report"
	@echo "  make lint          - Run code linters"
	@echo "  make format        - Format code"
	@echo "  make audit         - Run full code audit"
	@echo "  make clean         - Clean temporary files"

install:
	uv sync --all-groups

run:
	cd src && uv run uvicorn llm_intelligence.main:app --host 0.0.0.0 --port 8001

dev:
	cd src && uv run uvicorn llm_intelligence.main:app --host 0.0.0.0 --port 8001 --reload

test:
	uv run --group dev pytest

test-coverage:
	uv run --group dev pytest --cov=src/llm_intelligence --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	uv run --group dev ruff check src/
	uv run --group dev mypy src/

format:
	uv run --group dev ruff format src/
	uv run --group dev ruff check --fix src/

audit:
	bash code_audit.sh

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".llm_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned"
