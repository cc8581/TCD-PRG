param(
    [string]$InstallRoot = ".deps"
)
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$installPath = Join-Path $repoRoot $InstallRoot
New-Item -ItemType Directory -Force -Path $installPath | Out-Null

$dependencies = @(
    @{
        Name = "graspnet-baseline"
        Url = "https://github.com/graspnet/graspnet-baseline.git"
        Revision = "280c215129f759ed8649cb4e89fc5dfee55f4f80"
        Patch = (Join-Path $repoRoot "patches\graspnet_windows_int64.patch")
    },
    @{
        Name = "graspnetAPI"
        Url = "https://github.com/graspnet/graspnetAPI.git"
        Revision = "eb57dd2092d8dbe05312a29c3d0c22f3226efbfc"
        Patch = $null
    }
)

foreach ($dependency in $dependencies) {
    $destination = Join-Path $installPath $dependency.Name
    if (-not (Test-Path -LiteralPath $destination)) {
        git clone $dependency.Url $destination
    }
    $actual = git -C $destination remote get-url origin
    if ($actual -ne $dependency.Url) {
        throw "Unexpected remote for $destination: $actual"
    }
    git -C $destination fetch --tags origin
    git -C $destination checkout --detach $dependency.Revision
    if ($dependency.Patch) {
        git -C $destination apply --reverse --check $dependency.Patch 2>$null
        if ($LASTEXITCODE -ne 0) {
            git -C $destination apply --check $dependency.Patch
            git -C $destination apply $dependency.Patch
        }
    }
}

Write-Host "Third-party dependencies installed under $installPath"
