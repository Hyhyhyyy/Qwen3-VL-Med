param(
    [Parameter(Mandatory = $true)]
    [string]$TrainSource,
    [Parameter(Mandatory = $true)]
    [string]$TestSource,
    [Parameter(Mandatory = $true)]
    [string[]]$DestinationDataDirs
)

$ErrorActionPreference = 'Stop'
$sourcePrefix = '<RAW_WSI_DATA_DIR>/'
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Convert-Annotation([string]$path) {
    $raw = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)
    $occurrences = ([regex]::Matches($raw, [regex]::Escape($sourcePrefix))).Count
    if ($occurrences -eq 0) {
        throw "初版 JSON 中未找到预期图片路径前缀：$path"
    }
    return [pscustomobject]@{
        Text = $raw.Replace($sourcePrefix, 'wsi_train/')
        Replaced = $occurrences
    }
}

$train = Convert-Annotation $TrainSource
$test = Convert-Annotation $TestSource
$trainRecords = $train.Text | ConvertFrom-Json
$validTrainRecords = New-Object System.Collections.Generic.List[object]
$excludedTrainRecords = New-Object System.Collections.Generic.List[object]
for ($index = 0; $index -lt $trainRecords.Count; $index++) {
    $record = $trainRecords[$index]
    $assistantMessages = @($record.messages | Where-Object { $_.role -eq 'assistant' })
    $hasEmptyAssistant = $assistantMessages.Count -eq 0 -or @(
        $assistantMessages | Where-Object { [string]::IsNullOrWhiteSpace([string]$_.content) }
    ).Count -gt 0
    if ($hasEmptyAssistant) {
        $excludedTrainRecords.Add([pscustomobject]@{
            type = 'empty_assistant'
            source_record_index = $index + 1
            images = @($record.images)
            messages = @($record.messages)
        })
    } else {
        $validTrainRecords.Add($record)
    }
}
$validTrainText = $validTrainRecords | ConvertTo-Json -Compress -Depth 10
$excludedLines = @(
    $excludedTrainRecords | ForEach-Object { $_ | ConvertTo-Json -Compress -Depth 10 }
) -join "`n"
if ($excludedLines) {
    $excludedLines += "`n"
}

foreach ($destination in $DestinationDataDirs) {
    $resolved = [System.IO.Path]::GetFullPath($destination)
    [System.IO.Directory]::CreateDirectory($resolved) | Out-Null
    [System.IO.File]::WriteAllText(
        [System.IO.Path]::Combine($resolved, 'wsi_train.json'),
        $validTrainText,
        $utf8NoBom
    )
    [System.IO.File]::WriteAllText(
        [System.IO.Path]::Combine($resolved, 'wsi_train_source.json'),
        $train.Text,
        $utf8NoBom
    )
    [System.IO.File]::WriteAllText(
        [System.IO.Path]::Combine($resolved, 'excluded_records.jsonl'),
        $excludedLines,
        $utf8NoBom
    )
    [System.IO.File]::WriteAllText(
        [System.IO.Path]::Combine($resolved, 'wsi_test.json'),
        $test.Text,
        $utf8NoBom
    )
    Write-Output "已写入 $resolved（train=$($validTrainRecords.Count), excluded=$($excludedTrainRecords.Count), test=$($test.Replaced)）"
}
