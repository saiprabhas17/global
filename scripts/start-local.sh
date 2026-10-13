#!/usr/bin/env bash
set -euo pipefail
gmee_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$gmee_root"
gmee_prepare_models=false
gmee_open_browser=true
for argument in "$@"; do
    case "$argument" in
        --prepare-models) gmee_prepare_models=true ;;
        --no-browser) gmee_open_browser=false ;;
        *) echo "Usage: bash scripts/start-local.sh [--prepare-models] [--no-browser]" >&2; exit 2 ;;
    esac
done
if ! command -v docker >/dev/null 2>&1; then
    echo 'Install and start Docker Desktop (or Docker Engine with Compose v2), then rerun this launcher.' >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo 'Docker is not running. Start its engine and rerun this launcher.' >&2
    exit 1
fi
docker compose version >/dev/null
if [ ! -f .env ]; then
    if ! command -v openssl >/dev/null 2>&1; then
        echo 'OpenSSL is required to generate unique local credentials.' >&2
        exit 1
    fi
    (
        umask 077
        set -o noclobber
        {
            echo 'POSTGRES_USER=gmee'
            echo 'POSTGRES_DB=gmee'
            echo "POSTGRES_PASSWORD=$(openssl rand -hex 32)"
            echo "NEO4J_PASSWORD=$(openssl rand -hex 32)"
            echo "JWT_SECRET=$(openssl rand -hex 32)"
            echo 'CORS_ORIGINS=["http://localhost:3000","http://127.0.0.1:3000"]'
            echo 'ENABLE_SCHEDULER=false'
            echo 'HF_HUB_DISABLE_XET=1'
            echo 'OMP_NUM_THREADS=2'
            echo 'NUMBA_NUM_THREADS=2'
            echo 'TOKENIZERS_PARALLELISM=false'
        } > .env
    )
    echo 'Created private local configuration with unique credentials.'
else
    echo 'Using your existing .env and database volumes.'
fi
gmee_compose=(docker compose --env-file .env -f infra/docker-compose.yml -f infra/docker-compose.local.yml)
echo 'Building and starting GMEE. The first run downloads dependencies and may take several minutes.'
"${gmee_compose[@]}" up -d --build --wait --wait-timeout 300
if "$gmee_prepare_models"; then
    "${gmee_compose[@]}" exec -T backend python scripts/prepare_models.py
fi
"${gmee_compose[@]}" exec -T frontend wget -q -O /dev/null http://127.0.0.1/api/v1/health/ready
echo 'GMEE is ready: http://localhost:3000'
if "$gmee_open_browser"; then
    if command -v open >/dev/null 2>&1; then
        open http://localhost:3000 || true
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open http://localhost:3000 >/dev/null 2>&1 || true
    fi
fi
