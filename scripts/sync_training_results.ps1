# scripts/sync_training_results.ps1
#
# Pulls the GPU server's training log and per-epoch metrics down to this
# laptop every N seconds, so you can watch progress locally (in a text
# editor, or just `Get-Content -Wait` the log) without re-connecting over
# SSH each time. Does NOT pull checkpoint.pt by default -- that file is
# tens of MB and only needed if you actually want to resume/use the
# model locally, not just watch progress; pass -IncludeCheckpoint to
# also pull it.
#
# Usage (from the repo root, in PowerShell):
#   .\scripts\sync_training_results.ps1
#   .\scripts\sync_training_results.ps1 -IntervalSeconds 15
#   .\scripts\sync_training_results.ps1 -IncludeCheckpoint
#
# Stop with Ctrl+C at any time -- it only ever reads from the remote, so
# there's nothing to clean up.

param(
    [string]$KeyPath = "$HOME\.ssh\id_ed25519_drishti_gpu",
    [string]$RemoteHost = "utkarsh@172.16.192.12",
    [string]$RemoteDir = "~/drishti/checkpoints",
    [string]$LocalDir = "checkpoints_remote",
    [int]$IntervalSeconds = 20,
    [switch]$IncludeCheckpoint
)

if (-not (Test-Path $LocalDir)) {
    New-Item -ItemType Directory -Path $LocalDir | Out-Null
}

Write-Host "Syncing $RemoteHost`:$RemoteDir -> .\$LocalDir every $IntervalSeconds s. Ctrl+C to stop."
Write-Host ""

while ($true) {
    $timestamp = Get-Date -Format "HH:mm:ss"

    # Log + all per-epoch metrics JSON + the stats file -- small, cheap
    # to re-pull every poll.
    & scp -i $KeyPath -q "${RemoteHost}:${RemoteDir}/train.log" "$LocalDir\" 2>$null
    & scp -i $KeyPath -q "${RemoteHost}:${RemoteDir}/val_metrics_epoch*.json" "$LocalDir\" 2>$null
    & scp -i $KeyPath -q "${RemoteHost}:${RemoteDir}/channel_stats.json" "$LocalDir\" 2>$null

    if ($IncludeCheckpoint) {
        & scp -i $KeyPath -q "${RemoteHost}:${RemoteDir}/checkpoint.pt" "$LocalDir\" 2>$null
    }

    $latestMetric = Get-ChildItem "$LocalDir\val_metrics_epoch*.json" -ErrorAction SilentlyContinue |
        Sort-Object { [int]($_.BaseName -replace 'val_metrics_epoch', '') } -Descending |
        Select-Object -First 1

    if ($latestMetric) {
        $m = Get-Content $latestMetric.FullName | ConvertFrom-Json
        Write-Host "[$timestamp] latest: epoch $($m.epoch), val mIoU=$([math]::Round($m.miou, 4)), train_set=$($m.train_set_size), val_set=$($m.val_set_size)"
    } else {
        Write-Host "[$timestamp] no metrics yet -- still on the first epoch, or training hasn't started"
    }

    if (Test-Path "$LocalDir\train.log") {
        $lastLine = Get-Content "$LocalDir\train.log" -Tail 1
        Write-Host "            log: $lastLine"
    }

    Start-Sleep -Seconds $IntervalSeconds
}
