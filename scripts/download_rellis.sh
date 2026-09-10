#!/usr/bin/env bash
# scripts/download_rellis.sh
#
# Ticket #3 — pull RELLIS-3D sequence 00004's KITTI-format point clouds
# into Google Drive from Colab.
#
# IMPORTANT, confirmed against the official repo (2026-09-10): the
# KITTI-format data (what perception/rellis_loader.py reads) is NOT
# published per-sequence. It ships as one combined Google Drive archive
# across all 5 sequences -- 14GB for the Ouster OS1-64 stream, 5.58GB for
# the Velodyne Ultra Puck stream. There is a *separate*, per-sequence
# "synced" ROS bag download (~7GB for seq 00004) but that is rosbag
# format, not the .bin files this loader expects -- do not download that
# one for this purpose, it needs bag-extraction tooling this repo doesn't
# have yet.
#
# So the real plan is: download the 14GB combined Ouster archive (or the
# 5.58GB Velodyne one, if you'd rather start with the lighter sensor),
# extract ONLY the 00004/ subfolder, and delete the rest -- you do not
# need to keep all 5 sequences on Drive.
#
# Usage (from a Colab cell, after mounting Drive):
#   !pip install -q gdown
#   !bash scripts/download_rellis.sh /content/drive/MyDrive/drishti_data
#   !bash scripts/download_rellis.sh /content/drive/MyDrive/drishti_data vel   # Velodyne instead

set -euo pipefail

DEST_ROOT="${1:?Usage: download_rellis.sh <dest_root_dir> [os1|vel]}"
STREAM="${2:-os1}"
SEQ="00004"
DEST="${DEST_ROOT}/rellis_raw"
FINAL="${DEST_ROOT}/rellis/${SEQ}"

mkdir -p "${DEST}" "${FINAL}"

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown not found -- install it first: pip install gdown" >&2
  exit 1
fi

if [[ "${STREAM}" == "os1" ]]; then
  ARCHIVE_ID="1lDSVRf_kZrD0zHHMsKJ0V1GN9QATR4wH"      # Ouster LiDAR, SemanticKITTI format, 14GB, all 5 sequences
  LABELS_ID="12bsblHXtob60KrjV7lGXUQTdC5PhV8Er"        # Ouster annotations, 174MB
  POSES_ID="1V3PT_NJhA41N7TBLp5AbW31d0ztQDQOX"          # Ouster poses, 174MB
  CLOUD_DIRNAME="os1_cloud_node_kitti_bin"
  LABEL_DIRNAME="os1_cloud_node_semantickitti_label_id"
elif [[ "${STREAM}" == "vel" ]]; then
  ARCHIVE_ID="1PiQgPQtJJZIpXumuHSig5Y6kxhAzz1cz"        # Velodyne LiDAR, SemanticKITTI format, 5.58GB, all 5 sequences
  LABELS_ID="1n-9FkpiH4QUP7n0PnQBp-s7nzbSzmxp8"          # Velodyne annotations, 143.6MB
  POSES_ID=""                                            # no separate Velodyne poses link found -- verify on the repo page
  CLOUD_DIRNAME="vel_cloud_node_kitti_bin"
  LABEL_DIRNAME="vel_cloud_node_semantickitti_label_id"
else
  echo "Unknown stream '${STREAM}' -- expected 'os1' or 'vel'." >&2
  exit 1
fi

echo "Downloading ${STREAM} archive (this is the combined all-sequences file, 5.6-14GB)..."
gdown --id "${ARCHIVE_ID}" -O "${DEST}/${STREAM}_kitti_format.zip"

echo "Downloading labels..."
gdown --id "${LABELS_ID}" -O "${DEST}/${STREAM}_labels.zip"

if [[ -n "${POSES_ID}" ]]; then
  echo "Downloading poses..."
  gdown --id "${POSES_ID}" -O "${DEST}/${STREAM}_poses.zip"
fi

echo "Extracting only sequence ${SEQ} from each archive..."
# CONFIRMED against a real download (2026-09-10): the archives' internal
# paths are prefixed with "Rellis-3D/", e.g. "Rellis-3D/00004/os1_cloud_
# node_kitti_bin/000000.bin" -- NOT "00004/..." directly at the zip root
# as originally assumed here. Without the prefix, unzip's pattern silently
# matches nothing and the script "succeeds" with an empty FINAL directory.
ARCHIVE_PREFIX="Rellis-3D"
unzip -q "${DEST}/${STREAM}_kitti_format.zip" "${ARCHIVE_PREFIX}/${SEQ}/*" -d "${DEST}/extract_cloud" || \
  { echo "If the archive's internal paths don't start with '${ARCHIVE_PREFIX}/${SEQ}/', inspect it with 'unzip -l' and adjust the pattern above."; exit 1; }
unzip -q "${DEST}/${STREAM}_labels.zip" "${ARCHIVE_PREFIX}/${SEQ}/*" -d "${DEST}/extract_labels" || true
[[ -n "${POSES_ID}" ]] && unzip -q "${DEST}/${STREAM}_poses.zip" "${ARCHIVE_PREFIX}/${SEQ}/*" -d "${DEST}/extract_poses" || true

cp -r "${DEST}/extract_cloud/${ARCHIVE_PREFIX}/${SEQ}/." "${FINAL}/" 2>/dev/null || true
cp -r "${DEST}/extract_labels/${ARCHIVE_PREFIX}/${SEQ}/." "${FINAL}/" 2>/dev/null || true
cp -r "${DEST}/extract_poses/${ARCHIVE_PREFIX}/${SEQ}/." "${FINAL}/" 2>/dev/null || true

echo "Once this settles, delete ${DEST} (the full-archive downloads) --"
echo "you only need ${FINAL} going forward. That's where the loader expects:"
echo "  ${FINAL}/${CLOUD_DIRNAME}/"
echo "  ${FINAL}/${LABEL_DIRNAME}/"
echo "  ${FINAL}/poses.txt"
echo ""
echo "VERIFY after extraction: 'unzip -l' each archive first if the extraction"
echo "step above found nothing -- the internal folder-naming convention inside"
echo "these specific Drive archives was not directly inspected while writing"
echo "this script, only inferred from the repo's documented directory layout."
