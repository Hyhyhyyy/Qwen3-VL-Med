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
        '.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.onnx',
        '.dcm', '.dicom', '.svs', '.ndpi', '.mrxs', '.tif', '.tiff',
        '.jpg', '.jpeg', '.png', '.jsonl', '.csv', '.tsv', '.xlsx',
        '.xls', '.parquet', '.zip', '.7z', '.rar', '.tar', '.gz',
        '.gpg', '.age', '.pem', '.key'
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
    $disclosedDataScalePattern = '(?:共有|剩余|训练集|测试集)[^\r\n]{0,40}\d+\s*(?:条|例|张)'
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
