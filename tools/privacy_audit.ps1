$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).ProviderPath
Push-Location $repoRoot
try {
    $listed = & git ls-files --cached --others --exclude-standard
    if ($LASTEXITCODE -ne 0) { throw 'git ls-files failed' }

    $files = @($listed | Where-Object { $_ -and -not $_.StartsWith('.git/') })
    $errors = New-Object System.Collections.Generic.List[string]
    $allowedDataTemplates = @(
        'docs/environment_config.csv',
        'docs/metrics/metric_dictionary.csv',
        'docs/metrics/r04_r07_pareto_template.csv',
        'docs/metrics/r04_r07_split_template.csv'
    )

    $blockedExtensions = @(
        '.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.onnx', '.gguf', '.h5', '.hdf5',
        '.npz', '.npy',
        '.dcm', '.dicom', '.svs', '.ndpi', '.mrxs', '.tif', '.tiff',
        '.jpg', '.jpeg', '.png', '.jsonl', '.csv', '.tsv', '.xlsx',
        '.xls', '.parquet', '.zip', '.7z', '.rar', '.tar', '.gz',
        '.gpg', '.age', '.pem', '.key', '.zst', '.xz', '.bz2'
    )
    $textExtensions = @(
        '.py', '.ps1', '.sh', '.yaml', '.yml', '.json', '.md', '.txt',
        '.csv', '.tsv', '.toml', '.ini', '.cfg', '.in', '.jinja', '.xml', '.svg'
    )
    $maxBytes = 5MB

    $privateKeyPattern = 'BEGIN' + '[ A-Z]*PRIVATE KEY'
    $internalMountPattern = '/course' + '[0-9]{2,}'
    $cloudHostPattern = 'px-' + 'cloud[0-9]*'
    $providerPattern = 'mat' + 'pool'
    $rootEndpointPattern = 'root' + '@'
    $windowsUserPattern = '[A-Za-z]:' + '\\Users\\'
    $openAiKeyPattern = 's' + 'k-(?:proj-|svcacct-)?[A-Za-z0-9_-]{16,}'
    $credentialUrlPattern = 'https?://[^\s/:]+:[^\s/@]+@'
    $legacyPrivateCountPattern = '\b(?:19' + '082|18' + '426|6' + '56|3' + '27)\b'
    # Build CJK phrases from code points so Windows PowerShell 5.1 cannot
    # corrupt a UTF-8 source file before the regex engine sees it.
    $scalePrefixes = @(
        ([string][char]0x5171 + [char]0x6709),
        ([string][char]0x5269 + [char]0x4F59),
        ([string][char]0x8BAD + [char]0x7EC3 + [char]0x96C6),
        ([string][char]0x6D4B + [char]0x8BD5 + [char]0x96C6)
    )
    $scaleUnits = @([string][char]0x6761, [string][char]0x4F8B, [string][char]0x5F20)
    $disclosedDataScalePattern = '(?:' + (($scalePrefixes | ForEach-Object { [regex]::Escape($_) }) -join '|') + ')[^\r\n]{0,40}\d+\s*(?:' + (($scaleUnits | ForEach-Object { [regex]::Escape($_) }) -join '|') + ')'
    $gitLfsPointerPattern = 'version\s+https://git-lfs\.github\.com/spec/v1'
    $patterns = @(
        $privateKeyPattern,
        'gh[opusr]_[A-Za-z0-9_]{20,}',
        'AKIA[0-9A-Z]{16}',
        $openAiKeyPattern,
        $credentialUrlPattern,
        $internalMountPattern,
        $cloudHostPattern,
        $providerPattern,
        $rootEndpointPattern,
        $windowsUserPattern,
        $legacyPrivateCountPattern,
        $disclosedDataScalePattern,
        $gitLfsPointerPattern,
        '\b1[3-9][0-9]{9}\b',
        '\b[0-9]{17}[0-9Xx]\b',
        '(MRN|medical_record_number|accession_number)[\s"'']*[:=][\s"'']*[A-Za-z0-9-]{4,}'
    )

    foreach ($relative in $files) {
        if ($relative.Contains('\')) {
            $errors.Add("non-portable backslash in Git path: $relative")
        }
        $path = Join-Path $repoRoot $relative
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { continue }
        $item = Get-Item -LiteralPath $path
        $extension = $item.Extension.ToLowerInvariant()
        $normalizedRelative = $relative.Replace('\', '/')
        $leaf = [System.IO.Path]::GetFileName($normalizedRelative).ToLowerInvariant()
        if ($leaf -match '(?:adapter_model|pytorch_model|model-[0-9]+-of-[0-9]+|consolidated|optimizer|zero_pp_rank)' -and $extension -notin @('.py','.md','.txt')) {
            $errors.Add("weight-like filename: $relative")
        }
        if (($blockedExtensions -contains $extension) -and ($allowedDataTemplates -notcontains $normalizedRelative)) {
            $errors.Add("blocked extension: $relative")
        }
        if ($item.Length -gt $maxBytes) {
            $errors.Add("file exceeds 5 MiB: $relative ($($item.Length) bytes)")
        }
        if ($textExtensions -contains $extension) {
            $content = Get-Content -Raw -LiteralPath $path
            foreach ($pattern in $patterns) {
                if ($content -match $pattern) {
                    $errors.Add("sensitive pattern [$pattern]: $relative")
                }
            }
        }
    }

    if ($errors.Count -gt 0) {
        $errors | Sort-Object -Unique | ForEach-Object { Write-Error $_ }
        exit 1
    }

    Write-Output "PRIVACY_AUDIT_OK files=$($files.Count) max_file_bytes=$maxBytes"
}
finally {
    Pop-Location
}
