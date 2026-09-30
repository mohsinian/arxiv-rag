-include .env
export DATABASE_URL

.PHONY: help up-db up down logs psql test test-integration lint format ui db-shell smoke ask ask-cli ingest reindex

CATEGORIES ?= cs.AI,cs.LG,cs.CL
LIMIT ?= 1000

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

up-db: ## Start the Postgres + pgvector container (waits until healthy)
	docker compose up -d --wait db

up: ## Build and start the full stack (db + api)
	docker compose up -d --build

down: ## Stop the stack (data volume is kept)
	docker compose down

ingest: ## Fetch newest papers (CATEGORIES=cs.AI,cs.LG,cs.CL LIMIT=1000)
	docker compose run --rm api arxiv-rag ingest --categories $(CATEGORIES) --limit $(LIMIT)

reindex: ## Embed all new/changed papers (CPU: ~20 min per 1,000)
	docker compose run --rm api arxiv-rag reindex

ask: ## Ask the RUNNING api (warm model, no reload), e.g. make ask QUESTION="What is X?"
	@test -n "$(QUESTION)" || (echo 'Usage: make ask QUESTION="your question"' && exit 2)
	@docker compose exec -T api arxiv-rag ask --api "$(QUESTION)" || \
		(echo 'API not reachable — run `make up` first, or use `make ask-cli` for the standalone path' && exit 2)

ask-cli: ## Ask via a one-off container (loads the model; nothing else needs to run)
	@test -n "$(QUESTION)" || (echo 'Usage: make ask-cli QUESTION="your question"' && exit 2)
	docker compose run --rm api arxiv-rag ask "$(QUESTION)"

smoke: ## Verify the composed API is serving
	curl -fsS http://localhost:$${API_PORT:-8000}/health && echo

logs: ## Tail database logs
	docker compose logs -f db

psql: ## Open psql inside the db container
	docker compose exec db psql -U $${POSTGRES_USER:-arxiv_rag} -d $${POSTGRES_DB:-arxiv_rag}

ui: ## Start Adminer (web DB UI) at http://localhost:8081
	docker compose --profile ui up -d adminer
	@echo "Adminer: http://localhost:8081  (server: db, user/password/db: arxiv_rag)"

db-shell: ## Terminal SQL shell on the host, no container needed (pgcli)
	.venv/bin/pgcli "$$DATABASE_URL"

test: ## Run unit tests (no Docker needed)
	.venv/bin/pytest

test-integration: ## Run integration tests (needs `make up-db` first)
	.venv/bin/pytest -m integration

lint: ## Ruff lint + format check
	.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests

format: ## Apply ruff formatting
	.venv/bin/ruff format src tests
