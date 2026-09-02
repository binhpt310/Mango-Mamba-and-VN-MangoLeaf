param(
    [string]$AdbPath = 'C:\Users\Admin\AppData\Local\Android\Sdk\platform-tools\adb.exe',
    [ValidateRange(1, 1000)]
    [int]$ImagesPerClass = 83,
    [string]$OutputRoot = "$PSScriptRoot\..\artifacts\device_throughput_$ImagesPerClass`_per_class",
    [string]$DeviceBasePath = '/sdcard/Android/data/com.mangomamba.benchmark/files/device_throughput',
    [int]$TimeoutSeconds = 900
)

$prepareScript = Join-Path $PSScriptRoot 'prepare_vn_mangoleaf_device_batch.ps1'
& $prepareScript -ImagesPerClass $ImagesPerClass -OutputRoot $OutputRoot -DeviceBasePath $DeviceBasePath
if ($LASTEXITCODE -ne 0) { throw 'Failed to prepare the deterministic device batch.' }

$expectedImages = $ImagesPerClass * 6
$manifestRemote = "$DeviceBasePath/device_manifest.tsv"
$resultsRemote = '/sdcard/Android/data/com.mangomamba.benchmark/files/vn_mangoleaf_batch_results.csv'
$resultsLocal = Join-Path $OutputRoot 'results.csv'

& $AdbPath shell mkdir -p $DeviceBasePath
& $AdbPath push "$OutputRoot\." $DeviceBasePath
& $AdbPath shell am force-stop com.mangomamba.benchmark
& $AdbPath shell am start -n com.mangomamba.benchmark/.MainActivity --es test_batch_manifest_path $manifestRemote

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
do {
    Start-Sleep -Seconds 5
    & $AdbPath pull $resultsRemote $resultsLocal 2>$null | Out-Null
    $rowCount = if (Test-Path -LiteralPath $resultsLocal) {
        [Math]::Max(0, (Get-Content -LiteralPath $resultsLocal -Encoding utf8).Count - 1)
    } else {
        0
    }
    Write-Output "Completed $rowCount/$expectedImages measured inferences"
} while ($rowCount -lt $expectedImages -and (Get-Date) -lt $deadline)

if ($rowCount -ne $expectedImages) {
    throw "Timed out or incomplete batch: $rowCount/$expectedImages"
}

$rows = Import-Csv -LiteralPath $resultsLocal -Encoding utf8
$latencies = @($rows | ForEach-Object { [double]$_.latency_ms } | Sort-Object)
$sumMs = ($latencies | Measure-Object -Sum).Sum
$correct = @($rows | Where-Object { $_.expected -eq $_.predicted }).Count
$summary = [ordered]@{
    measured_at_local = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ssK')
    images = $rows.Count
    images_per_class = $ImagesPerClass
    correct = $correct
    accuracy_percent = [Math]::Round(100 * $correct / $rows.Count, 4)
    total_session_run_seconds = [Math]::Round($sumMs / 1000, 3)
    average_session_run_ms = [Math]::Round(($sumMs / $rows.Count), 2)
    p50_session_run_ms = [Math]::Round($latencies[[Math]::Ceiling($latencies.Count * 0.50) - 1], 2)
    p95_session_run_ms = [Math]::Round($latencies[[Math]::Ceiling($latencies.Count * 0.95) - 1], 2)
    throughput_images_per_second = [Math]::Round(($rows.Count * 1000) / $sumMs, 3)
    manifest_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $OutputRoot 'device_manifest.tsv')).Hash
    results_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $resultsLocal).Hash
}
$summary | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputRoot 'throughput_summary.json') -Encoding utf8
$summary | ConvertTo-Json
