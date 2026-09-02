param(
    [Parameter(Mandatory)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$ModelPath,

    [Parameter(Mandatory)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$LabelsPath
)

$expectedLabels = @(
    'Anthracnose',
    'Dieback',
    'Gall Midge',
    'Healthy',
    'Powdery Mildew',
    'Red Rust',
    'Sooty Mold'
)

$labels = (Get-Content -Raw -Encoding utf8 -LiteralPath $LabelsPath | ConvertFrom-Json).class_names
if (@(Compare-Object -ReferenceObject $expectedLabels -DifferenceObject $labels -SyncWindow 0).Count -ne 0) {
    throw 'Labels must match the current seven supported classes in the exact order.'
}

$assetsRoot = Join-Path $PSScriptRoot '..\android\app\src\main\assets'
$targetModel = Join-Path $assetsRoot 'mango_mamba_7class.onnx'
$targetLabels = Join-Path $assetsRoot 'mango_mamba_7class_labels.json'

Copy-Item -LiteralPath $ModelPath -Destination $targetModel -Force
Copy-Item -LiteralPath $LabelsPath -Destination $targetLabels -Force

$modelHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $targetModel).Hash
$labelsHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $targetLabels).Hash

Write-Output "Replaced Android ONNX asset."
Write-Output "Model SHA-256:  $modelHash"
Write-Output "Labels SHA-256: $labelsHash"
Write-Output 'Next: build, install, run the device batch benchmark, then update ANDROID_MODEL_PROVENANCE.md.'
