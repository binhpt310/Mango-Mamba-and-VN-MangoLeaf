param(
    [string]$HostAlias = 'asus-ascent',
    [string]$RemoteRoot = '/home/admincsc/workspace/binh/reproduce_mm',
    [string]$ManifestPath = (Join-Path $PSScriptRoot '../docs/CHECKPOINT_SYNC_MANIFEST.json')
)
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
foreach ($entry in $manifest.entries) {
    if (-not $entry.remote_path) {
        Write-Warning "No checkpoint found in the recorded host inventory: $($entry.local_path)"
        continue
    }
    if (-not $entry.remote_path.StartsWith("$RemoteRoot/", [StringComparison]::Ordinal)) {
        throw "Remote checkpoint is outside RemoteRoot: $($entry.remote_path)"
    }
    $target = [IO.Path]::GetFullPath((Join-Path $workspace $entry.local_path))
    if (-not $target.StartsWith("$workspace$([IO.Path]::DirectorySeparatorChar)", [StringComparison]::OrdinalIgnoreCase)) {
        throw "Local checkpoint is outside workspace: $target"
    }
    if (Test-Path -LiteralPath $target) {
        if ((Get-Item -LiteralPath $target).Length -ne $entry.bytes -or
            (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $entry.sha256) {
            throw "Existing checkpoint differs: $target"
        }
        continue
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    $partial = "$target.part"
    & scp -o BatchMode=yes "${HostAlias}:$($entry.remote_path)" $partial
    if ($LASTEXITCODE -ne 0) { throw "scp failed: $($entry.remote_path)" }
    if ((Get-Item -LiteralPath $partial).Length -ne $entry.bytes -or
        (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $entry.sha256) {
        throw "Downloaded checkpoint failed verification: $partial"
    }
    Move-Item -LiteralPath $partial -Destination $target
    Write-Output "VERIFIED $($entry.local_path)"
}
