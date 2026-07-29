#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRIPT_DIR}/checkpoints}"
GROUNDING_FILE="${CHECKPOINT_DIR}/groundingdino_swinb_cogcoor.pth"
SAM_FILE="${CHECKPOINT_DIR}/sam_vit_h_4b8939.pth"

mkdir -p "${CHECKPOINT_DIR}"

download() {
  local url="$1"
  local output="$2"
  local partial="${output}.part"
  if [[ -f "${output}" ]]; then
    echo "Already present: ${output}"
    return
  fi
  curl --fail --location --continue-at - --output "${partial}" "${url}"
  mv "${partial}" "${output}"
}

download \
  "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth" \
  "${GROUNDING_FILE}"
download \
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
  "${SAM_FILE}"

verify() {
  local expected="$1"
  local file="$2"
  local actual
  actual="$(sha256sum "${file}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA-256 mismatch for ${file}" >&2
    echo "expected: ${expected}" >&2
    echo "actual:   ${actual}" >&2
    exit 1
  fi
  echo "Verified: ${file}"
}

verify \
  "46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7" \
  "${GROUNDING_FILE}"
verify \
  "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e" \
  "${SAM_FILE}"
