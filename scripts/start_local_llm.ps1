. (Join-Path $PSScriptRoot 'local_llm_common.ps1')
$endpoint = Get-LlmEndpoint
$receipt = Get-Content -LiteralPath (Get-LlmPath 'deployment.json') -Raw | ConvertFrom-Json
if ($receipt.config_sha256 -ne (Get-FileHash -LiteralPath $ConfigPath -Algorithm SHA256).Hash.ToLowerInvariant()) {
    throw 'Configuration changed since verification; rerun setup_local_llm.ps1 -Download (existing files are reused).'
}
foreach ($artifact in $LlmConfig.artifacts) { $null = Assert-LlmArtifact $artifact }
foreach ($file in $receipt.runtime_files) {
    if ((Get-FileHash -LiteralPath (Get-LlmPath $file.path) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) {
        throw "Runtime file changed: $($file.path)"
    }
}
$freeGiB = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
if ($freeGiB -lt $LlmConfig.minimum_free_ram_gib) {
    throw ('Only {0:N1} GiB RAM free; close other applications to free at least {1} GiB. Nothing started.' -f $freeGiB, $LlmConfig.minimum_free_ram_gib)
}
$listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Parse('127.0.0.1'), $endpoint.Port)
try { $listener.Start() } finally { $listener.Stop() }
$server = Get-LlmPath $receipt.server_path
$model = Get-LlmPath (@($LlmConfig.artifacts | Where-Object kind -eq 'model')[0].path)
$arguments = @('--model', ('"' + $model + '"'), '--alias', $LlmConfig.model_name,
    '--host', '127.0.0.1', '--port', $endpoint.Port, '--ctx-size', $LlmConfig.context_size,
    '--threads', $LlmConfig.threads, '--threads-batch', $LlmConfig.threads_batch,
    '--parallel', '1', '--n-gpu-layers', '0', '--offline', '--no-context-shift', '--log-verbosity', '2')
$process = Start-Process -FilePath $server -ArgumentList $arguments -WorkingDirectory $LlmRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Get-LlmPath 'server.stdout.log') -RedirectStandardError (Get-LlmPath 'server.stderr.log')
@{ pid = $process.Id; executable = $server; started_utc = $process.StartTime.ToUniversalTime().ToString('o');
   model = $model; arguments = $arguments; config_sha256 = $receipt.config_sha256 } |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Get-LlmPath 'server_process.json') -Encoding UTF8
Add-Type -AssemblyName System.Net.Http
$handler = [Net.Http.HttpClientHandler]::new()
$handler.UseProxy = $false
$client = [Net.Http.HttpClient]::new($handler)
$client.Timeout = [TimeSpan]::FromSeconds(2)
try {
    for ($attempt = 0; $attempt -lt 120; $attempt++) {
        $process.Refresh()
        if ($process.HasExited) { throw "Server exited. Read $LlmRoot\server.stderr.log; do not install missing system components blindly." }
        try {
            $response = $client.GetAsync("http://127.0.0.1:$($endpoint.Port)/health").GetAwaiter().GetResult()
            $healthy = $response.IsSuccessStatusCode
            $response.Dispose()
            if ($healthy) { Write-Host 'Server healthy. Next: uv run python scripts/probe_local_llm.py'; exit 0 }
        } catch { Write-Verbose 'Model is still loading.' }
        Start-Sleep -Seconds 1
    }
    throw 'Startup timed out. Server may still be loading; inspect logs. stop_local_llm.ps1 stops only this process.'
} finally { $client.Dispose(); $handler.Dispose() }
