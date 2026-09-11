# scripts/sync_training_results.ps1
#
# Pulls the GPU server's training log and per-epoch metrics down to this
# laptop every N seconds, so you can watch progress locally without
# re-connecting over SSH each time. Reads training_log.jsonl (the
# comprehensive per-epoch format added alongside the class-weighting/
# augmentation changes: train loss, val loss, val mIoU, per-class IoU,
# LR, epoch time, best-so-far, and an overfitting flag) plus the raw
# train.log tail for real-time context within the current epoch.
#
# Does NOT pull checkpoint .pt files by default -- those are tens of MB
# each and only needed if you actually want to resume/use the model
# locally, not just watch progress; pass -IncludeCheckpoint to also
# pull best.pt.
#
# Usage (from the repo root, in PowerShell):
#   .\scripts\sync_training_results.ps1
#   .\scripts\sync_training_results.ps1 -RemoteDir "checkpoints_multi_v2" -IntervalSeconds 30
#   .\scripts\sync_training_results.ps1 -IncludeCheckpoint
#   .\scripts\sync_training_results.ps1 -Once          # single pull, no loop
#
# Stop with Ctrl+C at any time -- it only ever reads from the remote, so
# there's nothing to clean up.

param(
    [string]$RemoteHost = "drishti-gpu",
    [string]$RemoteDir = "checkpoints_multi_v2",
    [string]$LocalDir = "",
    [int]$IntervalSeconds = 20,
    [switch]$IncludeCheckpoint,
    [switch]$Once
)

if ([string]::IsNullOrEmpty($LocalDir)) {
    $LocalDir = "$RemoteDir`_remote"
}
if (-not (Test-Path $LocalDir)) {
    New-Item -ItemType Directory -Path $LocalDir | Out-Null
}

function Show-Progress {
    scp "${RemoteHost}:~/drishti/$RemoteDir/train.log" "$LocalDir\" 2>$null
    scp "${RemoteHost}:~/drishti/$RemoteDir/training_log.jsonl" "$LocalDir\" 2>$null
    if ($IncludeCheckpoint) {
        scp "${RemoteHost}:~/drishti/$RemoteDir/best.pt" "$LocalDir\" 2>$null
    }

    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "`n[$timestamp] ------------------------------------------------------------" -ForegroundColor DarkGray

    if (Test-Path "$LocalDir\training_log.jsonl") {
        $lines = Get-Content "$LocalDir\training_log.jsonl" | Where-Object { $_.Trim().Length -gt 0 }
        if ($lines.Count -gt 0) {
            $rows = $lines | ForEach-Object { $_ | ConvertFrom-Json }
            $latest = $rows | Select-Object -Last 1
            $best = $rows | Sort-Object val_miou -Descending | Select-Object -First 1

            Write-Host ("epoch {0}: train_loss={1}  val_loss={2}  val_miou={3}  lr={4}  ({5} min)" -f `
                $latest.epoch, `
                [math]::Round($latest.train_loss, 4), `
                [math]::Round($latest.val_loss, 4), `
                [math]::Round($latest.val_miou, 4), `
                ("{0:e2}" -f $latest.lr), `
                [math]::Round($latest.epoch_time_s / 60, 1))

            Write-Host ("best so far: epoch {0}, val_miou={1}" -f $best.epoch, [math]::Round($best.val_miou, 4)) -ForegroundColor Green

            if ($latest.overfitting_flag) {
                Write-Host "WARNING: val loss rising while train loss falls -- overfitting flagged" -ForegroundColor Red
            }

            # Highlight the two classes this fine-tune run specifically
            # targets (class 4 was stuck at 0.0 IoU, class 2 was weak)
            # -- see TRAINING_RESULTS.md's own "Known Limitations".
            $c4 = $latest.per_class_iou[4]
            $c2 = $latest.per_class_iou[2]
            $c4Str = if ($null -eq $c4) { "n/a" } else { [math]::Round($c4, 4) }
            $c2Str = if ($null -eq $c2) { "n/a" } else { [math]::Round($c2, 4) }
            Write-Host "watch: class 2 (CAUTION) IoU=$c2Str, class 4 (STATIC_OBSTACLE) IoU=$c4Str"
        } else {
            Write-Host "training_log.jsonl exists but no epoch has completed yet" -ForegroundColor Yellow
        }
    } else {
        Write-Host "no training_log.jsonl synced yet -- still on the first epoch, or training hasn't started" -ForegroundColor Yellow
    }

    if (Test-Path "$LocalDir\train.log") {
        $lastLine = Get-Content "$LocalDir\train.log" -Tail 1
        Write-Host "log: $lastLine" -ForegroundColor DarkGray
    }

    # scp against a file that doesn't exist yet (e.g. training_log.jsonl
    # before epoch 0 finishes) is an EXPECTED, already-handled case
    # above, not a real script failure -- clear its exit code so it
    # doesn't get reported as one.
    $global:LASTEXITCODE = 0
}

if ($Once) {
    Show-Progress
    return
}

$procCheck = ssh $RemoteHost "ps aux | grep perception.train | grep -v grep"
if ($procCheck) {
    Write-Host "Remote training process: RUNNING" -ForegroundColor Green
} else {
    Write-Host "Remote training process: NOT detected right now (may have finished, or not started yet)" -ForegroundColor Yellow
}

Write-Host "Syncing $RemoteHost`:~/drishti/$RemoteDir -> .\$LocalDir every $IntervalSeconds s. Ctrl+C to stop."

while ($true) {
    Show-Progress
    Start-Sleep -Seconds $IntervalSeconds
}
