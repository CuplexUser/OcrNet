<#
.SYNOPSIS
    Tests OcrNet and bundles the app into dist\OcrNet, ready to run, copy or zip.

.PARAMETER SelfContained
    Include the .NET runtime so the bundle runs on machines without the .NET 10 Desktop Runtime
    (adds roughly 150 MB).

.PARAMETER Zip
    Also create dist\OcrNet-<version>-win-x64.zip.

.PARAMETER SkipTests
    Skip the unit tests.

.EXAMPLE
    ./build.ps1
    ./build.ps1 -Zip
    ./build.ps1 -SelfContained -Zip
#>
param(
    [switch]$SelfContained,
    [switch]$Zip,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$app = Join-Path $root 'src\OcrNet.App\OcrNet.App.csproj'
$dist = Join-Path $root 'dist'
$bundle = Join-Path $dist 'OcrNet'

# Arguments are passed as an array: PowerShell would otherwise split "-p:Name=value" at the colon.
function Invoke-DotNet([string[]]$Arguments) {
    & dotnet @Arguments
    if ($LASTEXITCODE -ne 0) { throw "dotnet $($Arguments[0]) failed with exit code $LASTEXITCODE" }
}

if (-not $SkipTests) {
    Write-Host 'Running tests...' -ForegroundColor Cyan
    Invoke-DotNet @('test', (Join-Path $root 'tests\OcrNet.Core.Tests\OcrNet.Core.Tests.csproj'), '-c', 'Release', '--nologo', '-v', 'q')
}

$running = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($bundle, 'OrdinalIgnoreCase') }
if ($running) { throw "OcrNet is running from $bundle (process $($running.Id -join ', ')). Close it and run the script again." }

Write-Host "Publishing to $bundle..." -ForegroundColor Cyan
if (Test-Path $bundle) { Remove-Item $bundle -Recurse -Force }
Invoke-DotNet @('publish', $app, '-c', 'Release', '-r', 'win-x64', '-o', $bundle, '--nologo', '-v', 'q',
    "--self-contained:$($SelfContained.IsPresent.ToString().ToLowerInvariant())",
    '-p:DebugType=none', '-p:SatelliteResourceLanguages=en')
Copy-Item (Join-Path $root 'README.md') $bundle

$size = (Get-ChildItem $bundle -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("Bundle: {0} ({1:N1} MB){2}" -f $bundle, $size,
    $(if ($SelfContained) { ', self-contained' } else { ', needs the .NET 10 Desktop Runtime' })) -ForegroundColor Green

if ($Zip) {
    $version = (& dotnet msbuild $app -getProperty:Version).Trim()
    $suffix = if ($SelfContained) { '-self-contained' } else { '' }
    $archive = Join-Path $dist "OcrNet-$version-win-x64$suffix.zip"
    if (Test-Path $archive) { Remove-Item $archive -Force }
    Compress-Archive -Path $bundle -DestinationPath $archive
    Write-Host "Archive: $archive" -ForegroundColor Green
}
