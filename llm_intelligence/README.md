# LLM Intelligence Service

Standalone FastAPI microservice that generates LLM-powered insights from meeting transcripts.

## Quick Start

```bash
cp .env.example .env  # fill in GEMINI_API_KEY or configure OLLAMA_HOST
make install
make dev              # starts on port 8001
```

## API

- `GET /health` — service status
- `GET /skills` — list available insight skills
- `POST /insights` — generate an insight (`skill_name`, `transcript`, optional `model`)

## Skills

Add `.md` files to `skills/` with YAML frontmatter + prompt template using `{TRANSCRIPT}` placeholder.
