param([switch] $Download)
. (Join-Path $PSScriptRoot 'local_llm_common.ps1')

Write-Host "Project-local deployment: $LlmRoot"
Write-Host 'Qwen2.5-7B-Instruct Q4_K_M; CPU-only llama.cpp; no service, PATH or driver changes.'
Write-Host ('Total download: {0:N2} GB' -f (($LlmConfig.artifacts | Measure-Object size -Sum).Sum / 1e9))
foreach ($artifact in $LlmConfig.artifacts) {
    Write-Host ('{0}: {1:N0} bytes; SHA256 {2}' -f $artifact.path, $artifact.size, $artifact.sha256)
}
if (-not $Download) {
    Write-Host 'Plan only. To download and verify, run this script with -Download.'
    exit 0
}
$curl = (Get-Command curl.exe -ErrorAction Stop).Source
$drive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($LlmRoot))
if ($drive.AvailableFreeSpace -lt 7GB) { throw 'At least 7 GiB free disk space is required before setup.' }
New-Item -ItemType Directory -Path $LlmRoot -Force | Out-Null
foreach ($artifact in $LlmConfig.artifacts) {
    $destination = Get-LlmPath $artifact.path
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    if (-not (Test-Path -LiteralPath $destination)) {
        $partial = Get-LlmPath ($artifact.path + '.part')
        Write-Host "Downloading/resuming $($artifact.path)"
        & $curl --fail --location --proto '=https' --proto-redir '=https' --retry 4 --continue-at - --output $partial $artifact.url
        if ($LASTEXITCODE -ne 0) { throw 'Download failed; partial file retained for a resumed download.' }
        if ((Get-Item -LiteralPath $partial).Length -ne $artifact.size) { throw "Wrong download size: $partial" }
        if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant() -ne $artifact.sha256) {
            throw "Invalid SHA256: $partial. Inspect/remove this partial file manually before retrying."
        }
        Move-Item -LiteralPath $partial -Destination $destination
    }
    $null = Assert-LlmArtifact $artifact
}
$runtimeDir = Get-LlmPath ('runtime/' + $LlmConfig.runtime_tag)
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$archive = Get-LlmPath ($LlmConfig.artifacts | Where-Object kind -eq 'runtime').path
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead($archive)
try {
    foreach ($entry in $zip.Entries) {
        $target = [IO.Path]::GetFullPath((Join-Path $runtimeDir $entry.FullName))
        if (-not $target.StartsWith($runtimeDir + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Unsafe runtime ZIP entry.'
        }
    }
} finally { $zip.Dispose() }
Expand-Archive -LiteralPath $archive -DestinationPath $runtimeDir -Force
$servers = @(Get-ChildItem -LiteralPath $runtimeDir -Filter llama-server.exe -Recurse -File)
if ($servers.Count -ne 1) { throw 'Expected exactly one llama-server.exe in the verified archive.' }
$runtimeFiles = @(Get-ChildItem -LiteralPath $runtimeDir -Recurse -File | ForEach-Object {
    @{ path = $_.FullName.Substring($LlmRoot.Length + 1); sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
})
$receipt = @{
    ready = $true; verified_at_utc = [DateTime]::UtcNow.ToString('o'); runtime_tag = $LlmConfig.runtime_tag
    model_name = $LlmConfig.model_name; model_revision = $LlmConfig.model_revision
    config_sha256 = (Get-FileHash -LiteralPath $ConfigPath -Algorithm SHA256).Hash.ToLowerInvariant()
    server_path = $servers[0].FullName.Substring($LlmRoot.Length + 1)
    artifacts = @($LlmConfig.artifacts); runtime_files = $runtimeFiles
}
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Get-LlmPath 'deployment.json') -Encoding UTF8
Write-Host 'Downloaded, extracted and hash-verified. No executable has been started.'
Write-Host 'Next: .\scripts\start_local_llm.ps1'
