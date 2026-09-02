param(
    [string]$SourceRoot = 'D:\Mango-Mamba-backup\vnmangoleaf_not_splitted',
    [string]$OutputRoot = "$PSScriptRoot\..\artifacts\vn_mangoleaf_device_batch_100",
    [string]$DeviceBasePath = '/sdcard/Android/data/com.mangomamba.benchmark/files/vn_mangoleaf_batch_100',
    [switch]$IncludeHealthy,
    [string]$HealthySourceRoot = 'D:\Mango-Mamba-backup\mangomamba_android_build_cache_temp\root-build\test-images-20260802\vn\Healthy'
)

$classMap = [ordered]@{
    Anthracnose      = 'Anthracnose'
    Dieback          = 'Dieback'
    Gall_Midge       = 'Gall Midge'
    Powdery_Mildew   = 'Powdery Mildew'
    Red_Rust         = 'Red Rust'
    Sooty_Mold       = 'Sooty Mold'
}

if (Test-Path -LiteralPath $OutputRoot) {
    throw "OutputRoot already exists: $OutputRoot"
}

New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$manifestRows = [System.Collections.Generic.List[string]]::new()
$summary = [ordered]@{}

foreach ($entry in $classMap.GetEnumerator()) {
    $sourceClass = Join-Path $SourceRoot $entry.Key
    $targetClass = Join-Path $OutputRoot $entry.Value
    $images = Get-ChildItem -LiteralPath $sourceClass -File -Recurse -Include *.jpg,*.jpeg,*.png |
        Sort-Object FullName |
        Select-Object -First 100
    if ($images.Count -ne 100) {
        throw "Expected 100 images for $($entry.Key), found $($images.Count)"
    }

    New-Item -ItemType Directory -Path $targetClass | Out-Null
    foreach ($image in $images) {
        Copy-Item -LiteralPath $image.FullName -Destination (Join-Path $targetClass $image.Name)
        $devicePath = "$DeviceBasePath/$($entry.Value)/$($image.Name)"
        $manifestRows.Add("$($entry.Value)`t$devicePath")
    }
    $summary[$entry.Value] = $images.Count
}

if ($IncludeHealthy) {
    $healthyImages = Get-ChildItem -LiteralPath $HealthySourceRoot -File -Recurse -Include *.jpg,*.jpeg,*.png |
        Sort-Object FullName
    if ($healthyImages.Count -eq 0) {
        throw "No Healthy images found at $HealthySourceRoot"
    }

    $healthyTarget = Join-Path $OutputRoot 'Healthy'
    New-Item -ItemType Directory -Path $healthyTarget | Out-Null
    foreach ($image in $healthyImages) {
        Copy-Item -LiteralPath $image.FullName -Destination (Join-Path $healthyTarget $image.Name)
        $manifestRows.Add("Healthy`t$DeviceBasePath/Healthy/$($image.Name)")
    }
    $summary['Healthy'] = $healthyImages.Count
}

$manifestPath = Join-Path $OutputRoot 'device_manifest.tsv'
$manifestRows | Set-Content -LiteralPath $manifestPath -Encoding utf8
$summary | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputRoot 'selection_summary.json') -Encoding utf8
Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath |
    Select-Object Algorithm, Hash, Path |
    ConvertTo-Json |
    Set-Content -LiteralPath (Join-Path $OutputRoot 'manifest_sha256.json') -Encoding utf8

Write-Output "Prepared $($manifestRows.Count) images at $OutputRoot"
