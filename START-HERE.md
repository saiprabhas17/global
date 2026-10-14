# Open GMEE on your computer

The cloud workspace's `localhost` is separate from your computer's `localhost`.
Use a cloud preview/forwarded URL when your interface supplies one. Otherwise,
start this updated project on your own computer using the launchers below.

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and
start its engine. Extract the project download, then open a terminal in the
`GMEE` directory containing this file. Linux users can use Docker Engine with
Docker Compose v2. Use a current Compose release (the stack uses required env files).

## Windows PowerShell

With Docker Desktop running, double-click **Start-GMEE.bat** in the extracted
project. It opens the website after startup and keeps the console visible if
anything fails. You can also run the launcher manually:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-local.ps1
```

This bypass applies to that launcher process; it does not change your global
PowerShell policy. The launcher opens the browser after the stores/API are ready.

## macOS or Linux

```bash
bash scripts/start-local.sh
```

Then open **http://localhost:3000** on that computer. The first build downloads
the Python/Node dependencies and datastore images, so it can take several minutes.
Keep Docker running. The launchers create unique private credentials if `.env`
is absent and preserve existing configuration/volumes. Never share `.env`.

The desktop stack uses the Compose project name `gmee-local`. Its containers,
volumes and networks belong to that project, and Docker allocates available
network addresses. Older installations named `infra`, including their database
contents, remain separate. This desktop stack starts with its own empty databases;
it does not automatically import records from an older installation.

Local NLP additionally needs the model weights. To prepare them once:

```powershell
# Windows
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-local.ps1 -PrepareModels
```

```bash
# macOS / Linux
bash scripts/start-local.sh --prepare-models
```

The backend uses the tested Linux x86-64 CPU dependencies; Apple Silicon runs
that backend through Docker emulation. Allow sufficient memory for Neo4j and
the ML models. The initial corpus is empty; use an administrator's pipeline
controls after configuring sources to collect/process real articles.

If startup fails, inspect the bounded logs:

```bash
docker compose --env-file .env -f infra/docker-compose.yml -f infra/docker-compose.local.yml logs --tail 80
```

### PostgreSQL reports `password authentication failed for user "gmee"`

Docker keeps database volumes when you replace or re-extract the project. If
the new copy generates another `.env`, its new password can differ from the
password already stored in PostgreSQL. Keep the original `.env` when updating
the project. A healthy PostgreSQL container alone does not verify that the app
can authenticate.

To synchronize this project's local PostgreSQL login with the current `.env`
and start GMEE again, run this in the extracted project folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-local.ps1 -RepairPostgresPassword
```

This explicitly changes the configured PostgreSQL role's password using the
container's local socket. It preserves database contents and uses the current
configuration. Do not delete the database volume to fix a password mismatch.
If another service reports an authentication error afterward, restore its
original credentials from your earlier `.env` and rerun the launcher.

### PostgreSQL reports `role "gmee" does not exist`

An older launcher could reuse the `infra` project and its fixed `gmee_*`
containers. That database may have been initialized for a different application
or administrator, so changing the password of `gmee` cannot repair it.

Update `infra/docker-compose.local.yml` to the current version and run
`Start-GMEE.bat` normally. The `gmee-local` project creates its own PostgreSQL,
Neo4j and Redis stores using your current `.env`. It does not change the older
database's users or remove its volumes. Keep `.env` when updating this installation.

### Restore records from the older `gmee_v2` database

If the app starts but the corpus is empty after moving to `gmee-local`, the old
records remain in the original Docker volume. First confirm the source database
and counts. For the older `gmee_postgres` container initialized with administrator
`postgres`, use:

```powershell
docker exec gmee_postgres psql -U postgres -d gmee_v2 -c "SELECT count(*) FROM claims;"
docker exec gmee_postgres psql -U postgres -d gmee_v2 -c "SELECT * FROM alembic_version;"
```

With the latest recovery script and backend source present, run from the project
folder while Docker Desktop and the local app are running:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\restore-local-data.ps1
```

The script supports source schema `gmee03`/`gmee04`. It saves a custom-format
PostgreSQL backup under `local-backups/<timestamp>/`, restores into a **new,
uniquely named database**, builds the updated backend, and migrates only that copy.
It verifies claim/article counts before and after migration, then updates only
`POSTGRES_DB` in `.env` and restarts the app. Failed application startup restores
the previous configuration. No existing database or Docker volume is deleted.
The source container must remain running; avoid ingestion into the old database
during recovery because changing counts will stop the switch.

The local graph is rebuilt from PostgreSQL's saved articles, claims, entities,
and relationships without rerunning NLP. Graph-only data from an older Neo4j
installation is not imported. A graph failure is reported separately and leaves
the verified recovered PostgreSQL database selected. The graph rebuild remains
bounded at 20,000 articles, 50,000 claims, and 100,000 relationships.

Keep `local-backups` private: it includes database contents and a copy of the
previous `.env`. This directory is ignored by Git. Sign in using an account from
the recovered database; accounts created only in the previous local database
remain there. The old `gmee` database, original `gmee_v2`, and previous local
database all remain available.

For native hot reload, install the dependencies as described in [README](README.md),
start the backend on 8000, then run `npm run dev` from `frontend`. Its default
port is 3000; `/api` proxies to that backend. Binding to `0.0.0.0` enables a
workspace preview/forwarder but does not create a public URL by itself.
