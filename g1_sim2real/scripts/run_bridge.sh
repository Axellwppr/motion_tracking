#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${G1_BRIDGE_BUILD_DIR:-${ROOT_DIR}/build}"
BIN="${BUILD_DIR}/g1_udp_bridge"
CONFIG="${G1_BRIDGE_CONFIG:-${ROOT_DIR}/config/g1_bridge.yaml}"
NET="${G1_NET:-lo}"

if [[ ! -x "${BIN}" ]]; then
  echo "Bridge binary not found: ${BIN}" >&2
  echo "Run: bash ${ROOT_DIR}/scripts/build.sh" >&2
  exit 1
fi

exec "${BIN}" --net "${NET}" --config "${CONFIG}" "$@"

