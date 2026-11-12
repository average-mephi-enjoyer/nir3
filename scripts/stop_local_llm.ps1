. (Join-Path $PSScriptRoot 'local_llm_common.ps1')
$record = Get-Content -LiteralPath (Get-LlmPath 'server_process.json') -Raw | ConvertFrom-Json
$process = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
if (-not $process) { Write-Host 'Recorded server is already stopped.'; exit 0 }
if ($process.Path -ne $record.executable -or $process.StartTime.ToUniversalTime().ToString('o') -ne $record.started_utc) {
    throw 'PID was reused or process identity differs; refusing to stop another process.'
}
$expected = Get-LlmPath ((Get-Content -LiteralPath (Get-LlmPath 'deployment.json') -Raw | ConvertFrom-Json).server_path)
if ($process.Path -ne $expected) { throw 'Recorded process does not belong to this deployment.' }
Stop-Process -Id $process.Id
Write-Host 'Project-local server stopped.'
