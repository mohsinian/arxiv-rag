#!/bin/sh
# Every container start — API server or one-off CLI job (`docker compose run
# --rm api arxiv-rag ...`) — gets a migrated schema before doing anything else.
set -e
alembic upgrade head
exec "$@"
