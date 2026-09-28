-include .env
export DATABASE_URL

.PHONY: help up-db down logs psql test test-integration lint format ui db-shell

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

up-db: ## Start the Postgres + pgvector container (waits until healthy)
	docker compose up -d --wait db

down: ## Stop the stack (data volume is kept)
	docker compose down

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
