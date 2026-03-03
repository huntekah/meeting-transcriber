.PHONY: help run

help:
	@echo "Available commands:"
	@echo "  make run  - Start ASR (8000), LLM Intelligence (8001), and CLI frontend"

run:
	@echo "Starting ASR (8000), LLM Intelligence (8001), and CLI..."
	@set -e; \
		mkdir -p asr_service/logs llm_intelligence/logs; \
		(cd asr_service && $(MAKE) run > logs/asr_service.log 2>&1) & ASR_PID=$$!; \
		(cd llm_intelligence && $(MAKE) run > logs/llm_intelligence.log 2>&1) & LLM_PID=$$!; \
		trap 'kill $$ASR_PID $$LLM_PID' INT TERM EXIT; \
		cd asr_service && $(MAKE) run-cli; \
		wait $$ASR_PID $$LLM_PID
