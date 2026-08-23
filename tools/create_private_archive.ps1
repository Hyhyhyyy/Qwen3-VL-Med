[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDirectory,

    [Parameter(Mandatory = $true)]
    [string]$OutputArchive
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).ProviderPath
$source = (Resolve-Path -LiteralPath $SourceDirectory).ProviderPath
$output = [IO.Path]::GetFullPath($OutputArchive)

if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw 'SourceDirectory must be an existing directory.'
}
if ([IO.Path]::GetExtension($output).ToLowerInvariant() -ne '.7z') {
    throw 'OutputArchive must use the .7z extension.'
}
if ($output.StartsWith($repoRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Encrypted private archives must be written outside the Git repository.'
}
if (Test-Path -LiteralPath $output) {
    throw 'OutputArchive already exists; refusing to overwrite it.'
}

$sevenZip = Get-Command 7z -ErrorAction Stop
$outputParent = Split-Path -Parent $output
if (-not (Test-Path -LiteralPath $outputParent -PathType Container)) {
    New-Item -ItemType Directory -Path $outputParent | Out-Null
}

Write-Host '7-Zip will prompt for a passphrase. Use an approved password manager; never store the passphrase in Git, scripts, environment files, shell history, or chat.'
& $sevenZip.Source a -t7z -mhe=on -p -- $output $source
if ($LASTEXITCODE -ne 0) { throw "7-Zip archive creation failed with exit code $LASTEXITCODE." }

Write-Host 'Testing archive integrity. Enter the same passphrase when prompted.'
& $sevenZip.Source t -p -- $output
if ($LASTEXITCODE -ne 0) { throw "7-Zip archive verification failed with exit code $LASTEXITCODE." }

$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $output).Hash.ToLowerInvariant()
$hashFile = "$output.sha256"
[IO.File]::WriteAllText($hashFile, "$hash  $([IO.Path]::GetFileName($output))`n", [Text.UTF8Encoding]::new($false))
Write-Output "PRIVATE_ARCHIVE_OK archive=$output sha256_file=$hashFile"
