param(
    [int]$Gpus = 2,
    [ValidateSet("all", "perception", "grasp", "push")]
    [string]$Stage = "all",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Overrides
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    & python train.py --gpus $Gpus --stage $Stage @Overrides
    if ($LASTEXITCODE -ne 0) {
        throw "Distributed training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
