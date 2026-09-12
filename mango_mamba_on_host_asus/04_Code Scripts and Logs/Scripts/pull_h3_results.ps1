param(
    [string]$Destination = (Join-Path $PSScriptRoot '..\..\99_Archive\Incoming Mango Mamba Final V3 Follow-up')
)

$remote = 'asus:/home/admincsc/workspace/binh/reproduce_mm'
$destination = [IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $destination) { throw "Destination already exists: $destination" }
New-Item -ItemType Directory -Path $destination | Out-Null
& scp -r "$remote/results/h3_320_aligned_followup" $destination
& scp -r "$remote/logs/h3_320_aligned_followup" $destination
if ($LASTEXITCODE -ne 0) { throw 'scp failed' }
