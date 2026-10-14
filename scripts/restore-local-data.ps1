param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]*$')]
    [string]$SourceContainer = 'gmee_postgres',
    [ValidatePattern('^[a-z_][a-z0-9_]*$')]
    [string]$SourceDatabase = 'gmee_v2',
    [ValidatePattern('^[a-z_][a-z0-9_]*$')]
    [string]$SourceUser = 'postgres',
    [switch]$NoBrowser
)

# Copy a legacy database; never restore over an existing database or volume.
# Compatible with Windows PowerShell 5.1: binary dumps travel through docker cp,
# never through PowerShell's text pipeline or output redirection.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$gmeeRoot = Split-Path $PSScriptRoot -Parent
Set-Location $gmeeRoot
$envPath = Join-Path $gmeeRoot '.env'
if (-not (Test-Path $envPath)) { throw 'Run Start-GMEE.bat once before recovering data.' }
foreach ($name in @('POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD', 'COMPOSE_PROJECT_NAME')) {
    if ([Environment]::GetEnvironmentVariable($name, 'Process')) {
        throw "Clear the session environment variable $name before running recovery; it can override .env."
    }
}

function Invoke-Docker {
    param([string[]]$DockerArguments)
    & docker @DockerArguments
    if ($LASTEXITCODE -ne 0) { throw 'A Docker recovery step failed. See the error above; existing databases were retained.' }
}

$compose = @('compose', '--project-name', 'gmee-local', '--env-file', '.env', '-f', 'infra/docker-compose.yml', '-f', 'infra/docker-compose.local.yml')
$null = Invoke-Docker @('info', '--format', '{{.ServerVersion}}')
$target = ((Invoke-Docker ($compose + @('ps', '-q', 'postgres'))) | Out-String).Trim()
if (-not $target -or $target -match '\s') { throw 'Start the local app first; exactly one local PostgreSQL container must be running.' }
$sourceId = ((Invoke-Docker @('inspect', '--format', '{{.Id}}', $SourceContainer)) | Out-String).Trim()
$targetId = ((Invoke-Docker @('inspect', '--format', '{{.Id}}', $target)) | Out-String).Trim()
if ($sourceId -eq $targetId) { throw 'Source and destination containers must be different.' }

$summarySql = "SELECT json_build_object('claims', (SELECT count(*) FROM public.claims), 'articles', (SELECT count(*) FROM public.articles), 'version', (SELECT version_num FROM public.alembic_version))::text;"
$sourceSummary = ((Invoke-Docker @('exec', $SourceContainer, 'psql', '-X', '-v', 'ON_ERROR_STOP=1', '-U', $SourceUser, '-d', $SourceDatabase, '-Atc', $summarySql)) | Out-String).Trim() | ConvertFrom-Json
if ($sourceSummary.version -notin @('gmee03', 'gmee04')) { throw 'This recovery supports gmee03/gmee04 databases only. No changes were made.' }
if ($sourceSummary.claims -lt 1) { throw 'The source database has no claims. No changes were made.' }
Write-Host "Found $($sourceSummary.claims) claims and $($sourceSummary.articles) articles in $SourceDatabase."

$runId = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$newDatabase = 'gmee_recovered_' + $runId
$backupDirectory = Join-Path $gmeeRoot ('local-backups/' + $runId)
$null = New-Item -ItemType Directory -Path $backupDirectory
$originalEnv = [System.IO.File]::ReadAllText($envPath)
if ([regex]::Matches($originalEnv, '(?m)^POSTGRES_DB=').Count -ne 1) {
    throw 'Expected exactly one POSTGRES_DB= setting in .env. No database changes were made.'
}
$recoveryEnv = [regex]::Replace($originalEnv, '(?m)^POSTGRES_DB=[^\r\n]*', ('POSTGRES_DB=' + $newDatabase))
$utf8 = [System.Text.UTF8Encoding]::new($false)
# Protect backups even when this script alone was downloaded into an older
# checkout whose root .gitignore does not yet know about local-backups.
[System.IO.File]::WriteAllText((Join-Path $backupDirectory '.gitignore'), "*`n", $utf8)
$envBackup = Join-Path $backupDirectory 'previous.env'
Copy-Item -LiteralPath $envPath -Destination $envBackup
$recoveryEnvPath = Join-Path $backupDirectory 'recovered.env'
[System.IO.File]::WriteAllText($recoveryEnvPath, $recoveryEnv, $utf8)
$recoveryCompose = @('compose', '--project-name', 'gmee-local', '--env-file', $recoveryEnvPath, '-f', 'infra/docker-compose.yml', '-f', 'infra/docker-compose.local.yml')
$dumpPath = '/tmp/gmee-recovery-' + $runId + '.dump'
$backupFile = Join-Path $backupDirectory ($SourceDatabase + '.dump')

Write-Host 'Backing up the original database. This can take several minutes.'
$null = Invoke-Docker @('exec', $SourceContainer, 'pg_dump', '-U', $SourceUser, '-d', $SourceDatabase, '--format=custom', '--file', $dumpPath)
$null = Invoke-Docker @('cp', "${SourceContainer}:$dumpPath", $backupFile)
if ((Get-Item -LiteralPath $backupFile).Length -lt 5) { throw 'The backup file is empty. Recovery stopped.' }
$backupHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $backupFile).Hash
[System.IO.File]::WriteAllText((Join-Path $backupDirectory 'backup.sha256'), $backupHash, $utf8)
$null = Invoke-Docker @('cp', $backupFile, "${target}:$dumpPath")
$null = Invoke-Docker @('exec', $target, 'pg_restore', '--list', $dumpPath)

Write-Host "Restoring into a new database: $newDatabase"
# createdb deliberately fails if a database with this unique name exists.
$null = Invoke-Docker @('exec', $target, 'sh', '-c', ('exec createdb -U "$POSTGRES_USER" --template=template0 ' + $newDatabase))
$null = Invoke-Docker @('exec', $target, 'sh', '-c', ('exec pg_restore -U "$POSTGRES_USER" --dbname=' + $newDatabase + ' --no-owner --no-acl --exit-on-error --single-transaction ' + $dumpPath))
function Read-RecoveredSummary {
    $summarySql | & docker exec -i $target sh -c ('exec psql -X -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d ' + $newDatabase + ' -At')
    if ($LASTEXITCODE -ne 0) { throw 'Could not verify the recovered database.' }
}
$restoredSummary = ((Read-RecoveredSummary) | Out-String).Trim() | ConvertFrom-Json
if ($restoredSummary.claims -ne $sourceSummary.claims -or $restoredSummary.articles -ne $sourceSummary.articles) {
    throw 'Counts changed during backup or restoration. The app has NOT switched databases. Stop any old ingestion jobs and retry.'
}

Write-Host 'Building the updated backend and applying migrations to the recovered copy.'
$null = Invoke-Docker ($recoveryCompose + @('build', 'backend'))
$null = Invoke-Docker ($recoveryCompose + @('run', '--rm', '--no-deps', '-T', 'backend', 'alembic', 'upgrade', 'head'))
$migratedSummary = ((Read-RecoveredSummary) | Out-String).Trim() | ConvertFrom-Json
if ($migratedSummary.version -ne 'gmee04' -or $migratedSummary.claims -ne $sourceSummary.claims -or $migratedSummary.articles -ne $sourceSummary.articles) {
    throw 'Migration verification failed. The app has NOT switched databases; backups and both databases were retained.'
}
[System.IO.File]::WriteAllText((Join-Path $backupDirectory 'recovery.json'), (@{
    sourceContainer = $SourceContainer; sourceDatabase = $SourceDatabase; recoveredDatabase = $newDatabase
    claims = $migratedSummary.claims; articles = $migratedSummary.articles; version = $migratedSummary.version
    backupSha256 = $backupHash
} | ConvertTo-Json), $utf8)
$null = Invoke-Docker @('exec', $SourceContainer, 'rm', '-f', '--', $dumpPath)
$null = Invoke-Docker @('exec', $target, 'rm', '-f', '--', $dumpPath)

Write-Host 'Counts verified. Switching the app to the recovered database.'
# Only application containers are replaced. PostgreSQL stays running with all
# original databases intact. A later ordinary launch may refresh its env vars.
try {
    [System.IO.File]::WriteAllText($envPath, $recoveryEnv, $utf8)
    $null = Invoke-Docker ($compose + @('up', '-d', '--no-deps', '--force-recreate', '--wait', '--wait-timeout', '300', 'backend', 'frontend'))
    $ready = Invoke-RestMethod -Uri 'http://127.0.0.1:3000/api/v1/health/ready' -TimeoutSec 30
    if ($ready.status -ne 'ready') { throw 'The recovered app did not pass its readiness check.' }
} catch {
    Copy-Item -LiteralPath $envBackup -Destination $envPath -Force
    Write-Host 'Restored the previous .env. Restarting the previous app configuration.' -ForegroundColor Yellow
    & docker @compose up -d --no-deps --force-recreate --wait --wait-timeout 300 backend frontend
    if ($LASTEXITCODE -ne 0) { Write-Warning 'The previous app could not restart. Both databases and the backup remain intact.' }
    throw 'Recovered data was retained, but app startup failed. Previous configuration restored; inspect Docker logs before retrying.'
}

Write-Host 'Rebuilding the local graph from saved claims and relationships; no NLP rerun is needed.'
$graphScript = @'
import asyncio
import json
from app.db.postgres import async_session_maker, engine
from app.db.neo4j_client import neo4j_client
from app.services.evolution.neo4j_writer import Neo4jWriter

async def main():
    try:
        driver = await neo4j_client.get_driver()
        async with driver.session() as graph:
            # Index MERGE lookups before loading a five-figure corpus.
            for label, prop in (("Article", "id"), ("Claim", "id"), ("Source", "id"), ("Domain", "name")):
                result = await graph.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{prop})")
                await result.consume()
            result = await graph.run("CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.text, n.type)")
            await result.consume()
            result = await graph.run("CALL db.awaitIndexes(300)")
            await result.consume()
        async with async_session_maker() as db:
            result = await Neo4jWriter().sync_persisted_graph(db)
        print(json.dumps(result))
        if not result.get("success"):
            raise SystemExit(1)
    finally:
        await neo4j_client.close()
        await engine.dispose()

asyncio.run(main())
'@
$graphScript | & docker @compose exec -T backend python -
if ($LASTEXITCODE -ne 0) {
    throw 'Claims and articles were restored and the app is running, but graph rebuilding failed. Keep the backup and send the graph error above.'
}
Write-Host "Recovered $($migratedSummary.claims) claims and $($migratedSummary.articles) articles."
Write-Host "Backup and previous configuration: $backupDirectory"
Write-Host 'Keep that folder private: it includes your database and local credentials.'
Write-Host 'Your original database and the previous local database were retained. Sign in with an account from the recovered database.'
Write-Host 'GMEE is ready: http://localhost:3000'
if (-not $NoBrowser) { Start-Process 'http://localhost:3000' }
