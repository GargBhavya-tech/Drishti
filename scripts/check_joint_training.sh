#!/bin/bash
# scripts/check_joint_training.sh
#
# Quick status check for the joint multi-dataset training run
# (perception/train_joint.py, checkpoints_joint/) on drishti-gpu.
# Usage: bash scripts/check_joint_training.sh
#        (or just: ssh drishti-gpu "tail -30 ~/drishti/checkpoints_joint_train.log")

ssh drishti-gpu "
echo '=== Process status ==='
ps aux | grep train_joint | grep -v grep || echo '(not running -- either finished or not yet started)'
echo
echo '=== Last 30 log lines ==='
tail -30 ~/drishti/checkpoints_joint_train.log
echo
echo '=== Epoch summary so far ==='
grep -E '\[JOINT\] Epoch [0-9]+/[0-9]+:|MEAN val mIoU' ~/drishti/checkpoints_joint_train.log
echo
echo '=== GPU status ==='
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
"
