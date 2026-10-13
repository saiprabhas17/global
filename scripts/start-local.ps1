param(
    [switch]$PrepareModels,
    [switch]$NoBrowser,
    [switch]$RepairPostgresPassword
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$gmeeRoot = Split-Path $PSScriptRoot -Parent
Set-Location $gmeeRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Install Docker Desktop, start it, then run this launcher again: https://www.docker.com/products/docker-desktop/'
}
$null = & docker info 2>&1
if ($LASTEXITCODE -ne 0) { throw 'Docker is not running. Open Docker Desktop and wait until its engine is ready.' }
$null = & docker compose version 2>&1
if ($LASTEXITCODE -ne 0) { throw 'Docker Compose v2 is required. Update Docker Desktop.' }

$gmeeEnvFile = Join-Path $gmeeRoot '.env'
if (-not (Test-Path $gmeeEnvFile)) {
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $secrets = @()
        for ($index = 0; $index -lt 3; $index++) {
            $bytes = New-Object byte[] 32
            $rng.GetBytes($bytes)
            $secrets += (($bytes | ForEach-Object { $_.ToString('x2') }) -join '')
        }
        $settings = @(
            'POSTGRES_USER=gmee', 'POSTGRES_DB=gmee',
            ('POSTGRES_PASSWORD=' + $secrets[0]),
            ('NEO4J_PASSWORD=' + $secrets[1]),
            ('JWT_SECRET=' + $secrets[2]),
            'CORS_ORIGINS=["http://localhost:3000","http://127.0.0.1:3000"]',
            'ENABLE_SCHEDULER=false', 'HF_HUB_DISABLE_XET=1',
            'OMP_NUM_THREADS=2', 'NUMBA_NUM_THREADS=2', 'TOKENIZERS_PARALLELISM=false'
        )
        $utf8 = [System.Text.UTF8Encoding]::new($false)
        $stream = [System.IO.File]::Open($gmeeEnvFile, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
        try {
            $writer = [System.IO.StreamWriter]::new($stream, $utf8)
            try { $writer.WriteLine(($settings -join "`n")) } finally { $writer.Dispose() }
        } finally { $stream.Dispose() }
        Write-Host 'Created private local configuration with unique credentials.'
    } finally { $rng.Dispose() }
} else {
    Write-Host 'Using your existing .env and database volumes.'
}

$composeArgs = @('compose', '--env-file', '.env', '-f', 'infra/docker-compose.yml', '-f', 'infra/docker-compose.local.yml')
if ($RepairPostgresPassword) {
    Write-Host 'Starting PostgreSQL to repair the local login while preserving database contents.'
    & docker @composeArgs up -d --wait --wait-timeout 120 postgres
    if ($LASTEXITCODE -ne 0) { throw 'PostgreSQL could not start. Password repair was not attempted.' }
    # The official PostgreSQL image permits its container-local Unix socket.
    # Read credentials inside that container; never place passwords in command
    # arguments, host output, or a generated SQL file.
    $repairSql = @'
\getenv gmee_role POSTGRES_USER
\getenv gmee_password POSTGRES_PASSWORD
ALTER ROLE :"gmee_role" WITH PASSWORD :'gmee_password';
'@
    $repairSql | & docker @composeArgs exec -T postgres sh -c 'exec psql -X -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
    if ($LASTEXITCODE -ne 0) { throw 'PostgreSQL password repair failed. Database contents were retained.' }
    Write-Host 'PostgreSQL login now matches the current local configuration.'
    # Clear the failed container's restart backoff and old unhealthy status
    # before Compose waits for readiness again. Datastores stay running.
    & docker @composeArgs stop frontend backend
    if ($LASTEXITCODE -ne 0) { throw 'Password repaired, but the app containers could not be stopped. Restart Docker Desktop and rerun this launcher.' }
}
Write-Host 'Building and starting GMEE. The first run downloads dependencies and may take several minutes.'
& docker @composeArgs up -d --build --wait --wait-timeout 300
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Startup failed. Recent backend output follows:' -ForegroundColor Yellow
    & docker @composeArgs logs --no-color --tail 80 backend
    Write-Host 'If it says password authentication failed, run:' -ForegroundColor Yellow
    Write-Host 'powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-local.ps1 -RepairPostgresPassword'
    throw 'GMEE could not start. Review the backend error above. Existing configuration and database contents were retained.'
}
if ($PrepareModels) {
    Write-Host 'Preparing the local NLP models. This downloads model weights once.'
    & docker @composeArgs exec -T backend python scripts/prepare_models.py
    if ($LASTEXITCODE -ne 0) { throw 'Model download failed. The web app is running, but NLP models need network access. Rerun with -PrepareModels after restoring access.' }
}
$ready = Invoke-RestMethod -Uri 'http://127.0.0.1:3000/api/v1/health/ready' -TimeoutSec 15
if ($ready.status -ne 'ready') { throw 'The web app responded, but a datastore is not ready.' }
Write-Host 'GMEE is ready: http://localhost:3000'
if (-not $NoBrowser) { Start-Process 'http://localhost:3000' }
