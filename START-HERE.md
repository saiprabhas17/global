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

For native hot reload, install the dependencies as described in [README](README.md),
start the backend on 8000, then run `npm run dev` from `frontend`. Its default
port is 3000; `/api` proxies to that backend. Binding to `0.0.0.0` enables a
workspace preview/forwarder but does not create a public URL by itself.
