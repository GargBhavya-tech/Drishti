# scripts/export_from_gpu.ps1
#
# Full export: pulls EVERYTHING relevant off the GPU server (drishti-gpu)
# down to this laptop in one shot -- the fine-tune run's entire checkpoint
# directory (every epoch checkpoint .pt file + training_log.jsonl +
# train.log + channel_stats.json) AND every eval/out/ artifact (confusion
# matrices, the accuracy-by-distance and latency evidence JSONs, PNG
# figures).
#
# The remote training job runs detached (nohup + disown, launched earlier
# this session) -- it does NOT care whether this laptop is on, asleep, or
# closed. Only the GPU SERVER itself needs to stay powered on and
# networked for it to keep running. Run this script whenever you're back
# at this laptop -- during training, right after it finishes, or days
# later -- and it pulls whatever the current state is.
#
# Each run REPLACES the local copy of each target directory with the
# remote's current state (not a merge) -- so it's always an exact mirror
# of what's on the server at the moment you run it, safe to re-run any
# number of times.
#
# Usage (from the repo root, in PowerShell):
#   .\scripts\export_from_gpu.ps1                                # everything, including all checkpoint .pt files
#   .\scripts\export_from_gpu.ps1 -SkipCheckpoints               # logs + best.pt only -- skip the large per-epoch .pt files
#   .\scripts\export_from_gpu.ps1 -RemoteDir "checkpoints_multi" # export a different run's checkpoint directory instead

param(
    [string]$RemoteHost = "drishti-gpu",
    [string]$RemoteDir = "checkpoints_multi_v2",
    [switch]$SkipCheckpoints
)

$ErrorActionPreference = "Continue"

function Copy-RemoteDir {
    param([string]$RemotePath, [string]$LocalPath, [string]$Label)
    Write-Host "`n--- $Label ---" -ForegroundColor Cyan
    if (Test-Path $LocalPath) {
        Remove-Item -Recurse -Force $LocalPath
    }
    scp -r "${RemoteHost}:$RemotePath" "$LocalPath" 2>&1 | ForEach-Object { Write-Host $_ }
}

Write-Host "Checking remote training process status..." -ForegroundColor DarkGray
$procCheck = ssh $RemoteHost "ps aux | grep perception.train | grep -v grep"
if ($procCheck) {
    Write-Host "Remote training process: STILL RUNNING -- exporting current, possibly incomplete, state" -ForegroundColor Yellow
} else {
    Write-Host "Remote training process: NOT detected (finished, or not started) -- exporting final state" -ForegroundColor Green
}

if ($SkipCheckpoints) {
    $localLogDir = "${RemoteDir}_remote"
    Write-Host "`n--- $RemoteDir (logs + best.pt only, skipping per-epoch .pt files) ---" -ForegroundColor Cyan
    if (-not (Test-Path $localLogDir)) { New-Item -ItemType Directory -Path $localLogDir -Force | Out-Null }
    scp "${RemoteHost}:~/drishti/$RemoteDir/training_log.jsonl" "$localLogDir\" 2>&1 | ForEach-Object { Write-Host $_ }
    scp "${RemoteHost}:~/drishti/$RemoteDir/train.log" "$localLogDir\" 2>&1 | ForEach-Object { Write-Host $_ }
    scp "${RemoteHost}:~/drishti/$RemoteDir/channel_stats.json" "$localLogDir\" 2>&1 | ForEach-Object { Write-Host $_ }
    scp "${RemoteHost}:~/drishti/$RemoteDir/best.pt" "$localLogDir\" 2>&1 | ForEach-Object { Write-Host $_ }
} else {
    Copy-RemoteDir "~/drishti/$RemoteDir" "${RemoteDir}_remote" "$RemoteDir (full checkpoint directory -- every epoch .pt, logs, stats)"
}

Copy-RemoteDir "~/drishti/eval/out" "eval\out_remote" "eval/out (all evidence artifacts: confusion matrices, accuracy-by-distance, latency, PNG figures)"

Write-Host "`nDone. Local copies:" -ForegroundColor Green
if ($SkipCheckpoints) {
    Write-Host "  .\${RemoteDir}_remote\  (training_log.jsonl, train.log, channel_stats.json, best.pt)"
} else {
    Write-Host "  .\${RemoteDir}_remote\  (full checkpoint directory)"
}
Write-Host "  .\eval\out_remote\      (everything currently in the GPU's eval/out/)"
