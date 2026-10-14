# Offline orchestration checks, also run under Windows PowerShell 5.1 in CI.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$originalDirectory = Get-Location
$fixture = Join-Path ([System.IO.Path]::GetTempPath()) ('gmee-recovery-test-' + [guid]::NewGuid().ToString('N'))
$null = New-Item -ItemType Directory -Path (Join-Path $fixture 'scripts')
Copy-Item (Join-Path $root 'scripts/restore-local-data.ps1') (Join-Path $fixture 'scripts/restore-local-data.ps1')
$envFile = Join-Path $fixture '.env'
$originalEnv = "POSTGRES_DB=gmee`r`nPOSTGRES_USER=gmee`r`n# preserved comment`r`n"
$global:LASTEXITCODE = 0
$global:gmeeRecoveryCommands = @()
$global:gmeeRecoveryMode = 'success'
$global:gmeeRecoveryReads = 0
$global:gmeeRecoveryStarts = 0

function global:docker {
    $global:LASTEXITCODE = 0
    $command = $args -join ' '
    $global:gmeeRecoveryCommands += $command
    if ($args -contains 'inspect') {
        if ($args[-1] -eq 'gmee_postgres') { 'source-id' } else { 'target-id' }
    } elseif ($args -contains 'ps') {
        'target-id'
    } elseif ($args -contains '-Atc') {
        if ($global:gmeeRecoveryMode -eq 'unsupported') {
            '{"claims":10212,"articles":10543,"version":"unknown"}'
        } else { '{"claims":10212,"articles":10543,"version":"gmee03"}' }
    } elseif ($args -contains 'cp') {
        if ($args[-2] -like 'gmee_postgres:*') {
            [System.IO.File]::WriteAllBytes($args[-1], [byte[]](80,71,68,77,80,0,255,13,10))
        }
    } elseif ($command -match 'exec psql') {
        $null = $input | Out-String
        $global:gmeeRecoveryReads++
        $claims = 10212
        if ($global:gmeeRecoveryMode -eq 'count-mismatch') { $claims = 1 }
        $version = 'gmee03'
        if ($global:gmeeRecoveryReads -gt 1) { $version = 'gmee04' }
        '{"claims":' + $claims + ',"articles":10543,"version":"' + $version + '"}'
    } elseif ($command -match 'exec -T backend python') {
        $null = $input | Out-String
        if ($global:gmeeRecoveryMode -eq 'graph-failed') { $global:LASTEXITCODE = 1 }
    } elseif ($args -contains 'up') {
        $global:gmeeRecoveryStarts++
        if ($global:gmeeRecoveryMode -eq 'startup-failed' -and $global:gmeeRecoveryStarts -eq 1) { $global:LASTEXITCODE = 1 }
    } elseif ($command -match 'alembic upgrade') {
        if ($global:gmeeRecoveryMode -eq 'migration-failed') { $global:LASTEXITCODE = 1 }
    } elseif ($command -match 'exec pg_restore' -and $global:gmeeRecoveryMode -eq 'restore-failed') {
        $global:LASTEXITCODE = 1
    } elseif ($args -contains 'pg_dump' -and $global:gmeeRecoveryMode -eq 'backup-failed') {
        $global:LASTEXITCODE = 1
    }
}
function global:Invoke-RestMethod {
    param($Uri, $TimeoutSec)
    return @{ status = 'ready' }
}
function global:Start-Process { throw 'The test must not open a browser.' }
function Assert-Condition {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

try {
    foreach ($mode in @('success', 'unsupported', 'count-mismatch', 'migration-failed', 'restore-failed', 'backup-failed', 'startup-failed', 'graph-failed')) {
        $global:gmeeRecoveryMode = $mode
        $global:gmeeRecoveryCommands = @()
        $global:gmeeRecoveryReads = 0
        $global:gmeeRecoveryStarts = 0
        [System.IO.File]::WriteAllText($envFile, $originalEnv)
        $failure = $null
        try { & (Join-Path $fixture 'scripts/restore-local-data.ps1') -NoBrowser }
        catch { $failure = $_ }
        if ($mode -eq 'success') {
            if ($failure) { throw $failure }
        } else { Assert-Condition ($null -ne $failure) "Expected $mode to fail." }
        $currentEnv = [System.IO.File]::ReadAllText($envFile)
        if ($mode -in @('success', 'graph-failed')) {
            Assert-Condition ($currentEnv -match 'POSTGRES_DB=gmee_recovered_') "$mode must retain the verified restored database."
            Assert-Condition ($currentEnv -match '# preserved comment') 'Recovery must preserve other .env settings.'
            Assert-Condition ($global:gmeeRecoveryReads -eq 2) 'Verify counts both before and after migration.'
        } else {
            Assert-Condition ($currentEnv -ceq $originalEnv) "$mode must preserve the previous .env."
            if ($mode -eq 'startup-failed') {
                Assert-Condition ($global:gmeeRecoveryStarts -eq 2) 'Startup failure must restart the previous configuration.'
            } else { Assert-Condition ($global:gmeeRecoveryStarts -eq 0) "$mode must not switch the app." }
        }
        Assert-Condition (@($global:gmeeRecoveryCommands | Where-Object { $_ -match 'dropdb|DROP DATABASE|down|--clean' }).Count -eq 0) 'Recovery must never delete or overwrite an existing database.'
        if ($mode -eq 'success') {
            $steps = $global:gmeeRecoveryCommands -join "`n"
            Assert-Condition ($steps -match '(?s)pg_dump.*cp.*pg_restore --list.*createdb.*pg_restore.*alembic upgrade.*up .*backend frontend.*exec -T backend python') 'Backup/restore/migrate/switch/graph steps must stay ordered.'
        }
        Write-Output "PASS recovery scenario: $mode"
    }
} finally {
    Set-Location $originalDirectory
    Remove-Item -LiteralPath $fixture -Recurse -Force
    Remove-Item Function:\docker, Function:\Invoke-RestMethod, Function:\Start-Process
}
