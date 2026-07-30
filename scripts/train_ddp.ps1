param(
    [int]$Gpus = 2,
    [string]$Config = "configs/config.yaml",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Overrides
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    & python scripts/launch_ddp_windows.py --nproc-per-node $Gpus `
        --config $Config @Overrides
    if ($LASTEXITCODE -ne 0) {
        throw "Distributed training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
