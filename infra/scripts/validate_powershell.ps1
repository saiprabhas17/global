# Parse only the allowed verification scripts; never invoke an API or read .env.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$files = @('verify_endpoints.ps1', 'infra/scripts/read_only_sweep.ps1', 'infra/scripts/validate_powershell.ps1', 'scripts/start-local.ps1', 'scripts/restore-local-data.ps1', 'infra/tests/test_local_launcher.ps1', 'infra/tests/test_local_recovery.ps1')
$failures = 0
foreach ($relative in $files) {
    $tokens = $null
    $parseErrors = $null
    $null = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $root $relative), [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count -gt 0) {
        $failures++
        Write-Output "FAIL PowerShell syntax: $relative"
        foreach ($parseError in $parseErrors) { Write-Output $parseError.Message }
    }
    else { Write-Output "PASS PowerShell syntax: $relative" }
}
if ($failures -gt 0) { exit 1 }
exit 0
