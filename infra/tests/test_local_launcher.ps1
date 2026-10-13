# Exercise the launcher without Docker, network requests, or real credentials.
# The password repair itself is additionally checked against disposable PostgreSQL.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$originalDirectory = Get-Location
$fixture = Join-Path ([System.IO.Path]::GetTempPath()) ('gmee-launcher-' + [guid]::NewGuid().ToString('N'))
$null = New-Item -ItemType Directory -Path (Join-Path $fixture 'scripts')
Copy-Item (Join-Path $root 'scripts/start-local.ps1') (Join-Path $fixture 'scripts/start-local.ps1')
$envFile = Join-Path $fixture '.env'
[System.IO.File]::WriteAllText($envFile, "EXISTING_CONFIGURATION=preserve`n")
$originalConfiguration = [System.IO.File]::ReadAllText($envFile)
$launcher = Join-Path $fixture 'scripts/start-local.ps1'
$global:gmeeTestCommands = @()
$global:gmeeTestMode = 'success'
$global:LASTEXITCODE = 0

function global:docker {
    $global:gmeeTestCommands += ($args -join ' ')
    $global:LASTEXITCODE = 0
    if ($args -contains 'exec') {
        $null = $input | Out-String
        if ($global:gmeeTestMode -eq 'repair-failed') { $global:LASTEXITCODE = 1 }
    } elseif ($args -contains 'logs') {
        Write-Output 'Fixture backend error: password authentication failed'
    } elseif ($args -contains 'up') {
        if (($global:gmeeTestMode -eq 'startup-failed') -and ($args -contains '--build')) {
            $global:LASTEXITCODE = 1
        }
        if (($global:gmeeTestMode -eq 'store-failed') -and ($args -contains 'postgres')) {
            $global:LASTEXITCODE = 1
        }
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
    & $launcher -NoBrowser
    Assert-Condition (@($global:gmeeTestCommands | Where-Object { $_ -match ' exec ' }).Count -eq 0) 'Ordinary startup must not rotate credentials.'

    $global:gmeeTestCommands = @()
    & $launcher -NoBrowser -RepairPostgresPassword
    $repair = @($global:gmeeTestCommands | Where-Object { $_ -match ' exec ' })
    Assert-Condition ($repair.Count -eq 1) 'Explicit repair must execute once.'
    Assert-Condition (@($global:gmeeTestCommands | Where-Object { $_ -match ' stop frontend backend$' }).Count -eq 1) 'Repair must reset the app restart state without stopping datastores.'
    $stopIndex = [Array]::FindIndex([string[]]$global:gmeeTestCommands, [Predicate[string]]{ param($item) $item -match ' stop frontend backend$' })
    $startIndex = [Array]::FindIndex([string[]]$global:gmeeTestCommands, [Predicate[string]]{ param($item) $item -match ' up .*--build' })
    Assert-Condition ($stopIndex -lt $startIndex) 'Readiness must wait after the app restart state is reset.'

    foreach ($mode in @('startup-failed', 'store-failed', 'repair-failed')) {
        $global:gmeeTestMode = $mode
        $global:gmeeTestCommands = @()
        $failed = $false
        try {
            if ($mode -eq 'startup-failed') { & $launcher -NoBrowser }
            else { & $launcher -NoBrowser -RepairPostgresPassword }
        } catch { $failed = $true }
        Assert-Condition $failed ('Launcher must fail when ' + $mode)
        if ($mode -eq 'startup-failed') {
            Assert-Condition (@($global:gmeeTestCommands | Where-Object { $_ -match ' logs .*backend$' }).Count -eq 1) 'Startup failure must show backend diagnostics.'
            Assert-Condition (@($global:gmeeTestCommands | Where-Object { $_ -match ' exec ' }).Count -eq 0) 'Startup failure must not automatically rotate credentials.'
        } elseif ($mode -eq 'store-failed') {
            Assert-Condition (@($global:gmeeTestCommands | Where-Object { $_ -match ' exec ' }).Count -eq 0) 'Unavailable PostgreSQL must not trigger repair.'
        } else {
            Assert-Condition (@($global:gmeeTestCommands | Where-Object { $_ -match ' up .*--build' }).Count -eq 0) 'Failed repair must not claim app startup.'
        }
    }
    Assert-Condition ([System.IO.File]::ReadAllText($envFile) -ceq $originalConfiguration) 'The existing .env must be preserved.'
    Write-Output 'PASS: five launcher scenarios; explicit repair only, restart order, diagnostics, failure handling, and configuration preservation.'
} finally {
    Set-Location $originalDirectory
    Remove-Item -Recurse -Force $fixture
    Remove-Item Function:\docker, Function:\Invoke-RestMethod, Function:\Start-Process
}
