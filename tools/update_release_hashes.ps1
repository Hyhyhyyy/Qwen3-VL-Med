$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$manifest = Join-Path $repoRoot 'RELEASE_SHA256.txt'

Push-Location $repoRoot
try {
    $files = @(& git ls-files --cached --others --exclude-standard | Where-Object { $_ -and $_ -ne 'RELEASE_SHA256.txt' })
    if ($LASTEXITCODE -ne 0) { throw 'git ls-files failed' }

    $lines = foreach ($relative in $files) {
        $path = Join-Path $repoRoot $relative
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { continue }
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
        "$hash  $($relative.Replace('\','/'))"
    }
    $content = ($lines -join "`n") + "`n"
    [IO.File]::WriteAllText($manifest, $content, [Text.UTF8Encoding]::new($false))
    Write-Output "RELEASE_HASHES_UPDATED files=$($lines.Count)"
}
finally {
    Pop-Location
}
