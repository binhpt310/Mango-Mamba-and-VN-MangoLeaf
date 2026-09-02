param(
    [string]$SourceRoot = 'D:\Mango-Mamba-backup\mangomamba_android_build_cache_temp\root-build\test-images-20260802\vn\Healthy',
    [string]$OutputRoot = "$PSScriptRoot\..\artifacts\vn_mangoleaf_healthy_probe",
    [string]$DeviceBasePath = '/sdcard/Android/data/com.mangomamba.benchmark/files/vn_mangoleaf_healthy_probe'
)

if (Test-Path -LiteralPath $OutputRoot) {
    throw "OutputRoot already exists: $OutputRoot"
}

$images = Get-ChildItem -LiteralPath $SourceRoot -File -Recurse -Include *.jpg,*.jpeg,*.png |
    Sort-Object FullName
if ($images.Count -eq 0) {
    throw "No Healthy images found at $SourceRoot"
}

New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$manifestRows = foreach ($image in $images) {
    Copy-Item -LiteralPath $image.FullName -Destination (Join-Path $OutputRoot $image.Name)
    "Healthy`t$DeviceBasePath/$($image.Name)"
}

$manifestPath = Join-Path $OutputRoot 'device_manifest.tsv'
$manifestRows | Set-Content -LiteralPath $manifestPath -Encoding utf8
[ordered]@{ Healthy = $images.Count } |
    ConvertTo-Json |
    Set-Content -LiteralPath (Join-Path $OutputRoot 'selection_summary.json') -Encoding utf8
Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath |
    Select-Object Algorithm, Hash, Path |
    ConvertTo-Json |
    Set-Content -LiteralPath (Join-Path $OutputRoot 'manifest_sha256.json') -Encoding utf8

Write-Output "Prepared $($images.Count) Healthy images at $OutputRoot"
