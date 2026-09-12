# scripts/check_joint_training.ps1
#
# Quick status check for the joint multi-dataset training run
# (perception/train_joint.py, checkpoints_joint/) on drishti-gpu.
# Usage: .\scripts\check_joint_training.ps1
#
# PowerShell-native version of check_joint_training.sh -- the .sh
# version fails when `bash` on this machine resolves to WSL's bash
# (common when WSL is installed), since WSL has its own separate SSH
# config/home directory, not C:\Users\<you>\.ssh\ where the real
# `drishti-gpu` Host entry and key actually live. This version calls
# Windows' own ssh.exe directly from PowerShell, which reads the
# correct config.

param(
    [string]$RemoteHost = "drishti-gpu"
)

Write-Host "=== Process status ===" -ForegroundColor Cyan
ssh $RemoteHost "ps aux | grep train_joint | grep -v grep || echo '(not running -- either finished or not yet started)'"

Write-Host "`n=== Last 30 log lines ===" -ForegroundColor Cyan
ssh $RemoteHost "tail -30 ~/drishti/checkpoints_joint_train.log"

Write-Host "`n=== Epoch summary so far ===" -ForegroundColor Cyan
ssh $RemoteHost "grep -E '\[JOINT\] Epoch [0-9]+/[0-9]+:|MEAN val mIoU' ~/drishti/checkpoints_joint_train.log"

Write-Host "`n=== GPU status ===" -ForegroundColor Cyan
ssh $RemoteHost "nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader"
