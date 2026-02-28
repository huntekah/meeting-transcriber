.PHONY: help install build-swift run dev run-cli run-insights run-all test test-coverage test-backend test-frontend clean lint format

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies using uv and compile Swift binaries"
	@echo "  make build-swift   - Compile ScreenCaptureKit Swift binary (macOS only)"
	@echo "  make run           - Start the ASR service (compiles Swift binary if needed)"
	@echo "  make dev           - Start with auto-reload (development mode, compiles Swift binary if needed)"
	@echo "  make run-cli       - Start the CLI frontend (compiles Swift binary if needed)"
	@echo "  make run-insights  - Start the LLM Intelligence service (port 8001)"
	@echo "  make run-all       - Start both ASR service and LLM Intelligence service"
	@echo "  make test          - Run all tests"
	@echo "  make test-coverage - Run tests with coverage report"
	@echo "  make test-backend  - Run backend tests only"
	@echo "  make test-frontend - Run frontend tests only"
	@echo "  make lint          - Run code linters"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean temporary files and compiled binaries"

build-swift:
	@if [ "$$(uname)" = "Darwin" ]; then \
		if ! [ -f scripts/screencapture_audio ]; then \
			echo "🔨 Compiling ScreenCaptureKit audio capture binary..."; \
			swiftc scripts/screencapture_audio.swift -o scripts/screencapture_audio; \
			echo "✅ ScreenCaptureKit binary compiled: scripts/screencapture_audio"; \
		else \
			echo "✅ ScreenCaptureKit binary already compiled"; \
		fi \
	else \
		echo "⏭️  ScreenCaptureKit only available on macOS, skipping..."; \
	fi

install: build-swift
	uv sync

run: build-swift
	cd src && uv run uvicorn asr_service.main:app --host 0.0.0.0 --port 8000

dev: build-swift
	cd src && uv run uvicorn asr_service.main:app --host 0.0.0.0 --port 8000 --reload

run-cli: build-swift
	cd src && uv run --group cli python -m cli_frontend

run-insights:
	cd ../llm_intelligence && make run

run-all: build-swift
	@echo "Starting ASR service (port 8000) and LLM Intelligence service (port 8001)..."
	@(cd src && uv run uvicorn asr_service.main:app --host 0.0.0.0 --port 8000) &
	@cd ../llm_intelligence && make run

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
	rm -f scripts/screencapture_audio
	@echo "✅ Cleaned compiled binaries"
