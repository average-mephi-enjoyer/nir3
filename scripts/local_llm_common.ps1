$ErrorActionPreference = 'Stop'
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$LlmRoot = [IO.Path]::GetFullPath((Join-Path $ProjectRoot 'local_llm'))
$ConfigPath = Join-Path $ProjectRoot 'configs/local_llm.json'
$LlmConfig = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
if ($LlmConfig.model_name -notmatch '^[A-Za-z0-9._-]+$') { throw 'Unsafe model alias.' }

function Get-LlmPath([string] $RelativePath) {
    if ([IO.Path]::IsPathRooted($RelativePath)) { throw 'Expected a relative project-local path.' }
    $resolved = [IO.Path]::GetFullPath((Join-Path $LlmRoot $RelativePath))
    $prefix = $LlmRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Path escapes local_llm directory.' }
    return $resolved
}

function Assert-LlmArtifact($Artifact) {
    $path = Get-LlmPath $Artifact.path
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing artifact: $path. Run setup_local_llm.ps1 -Download first." }
    if ((Get-Item -LiteralPath $path).Length -ne $Artifact.size) { throw "Wrong artifact size: $path" }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Artifact.sha256) { throw "SHA256 mismatch: $path. File has NOT been executed." }
    return $path
}

function Get-LlmEndpoint {
    $uri = [Uri]$LlmConfig.endpoint_url
    if ($uri.Scheme -ne 'http' -or $uri.Host -ne '127.0.0.1' -or $uri.UserInfo -or $uri.AbsolutePath -ne '/v1/chat/completions') {
        throw 'Only the configured literal loopback HTTP endpoint is supported.'
    }
    return $uri
}
