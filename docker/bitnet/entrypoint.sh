#!/usr/bin/env bash
set -euo pipefail

cd /opt/BitNet

BITNET_GGUF_REPO="${BITNET_GGUF_REPO:-microsoft/bitnet-b1.58-2B-4T-gguf}"
BITNET_MODEL_NAME="${BITNET_MODEL_NAME:-BitNet-b1.58-2B-4T}"
BITNET_MODEL_DIR="${BITNET_MODEL_DIR:-/data/models}"
BITNET_LOG_DIR="${BITNET_LOG_DIR:-/data/logs}"
BITNET_QUANT_TYPE="${BITNET_QUANT_TYPE:-i2_s}"
BITNET_USE_PRETUNED="${BITNET_USE_PRETUNED:-0}"
BITNET_THREADS="${BITNET_THREADS:-4}"
BITNET_CTX_SIZE="${BITNET_CTX_SIZE:-4096}"
BITNET_N_PREDICT="${BITNET_N_PREDICT:-256}"
BITNET_TEMPERATURE="${BITNET_TEMPERATURE:-0.2}"
BITNET_HOST="${BITNET_HOST:-0.0.0.0}"
BITNET_PORT="${BITNET_PORT:-8080}"
BITNET_CHAT_TEMPLATE="${BITNET_CHAT_TEMPLATE:-}"
BITNET_OVERRIDE_KV="${BITNET_OVERRIDE_KV:-}"
BITNET_ENABLE_CONT_BATCH="${BITNET_ENABLE_CONT_BATCH:-1}"

mkdir -p "${BITNET_MODEL_DIR}" "${BITNET_LOG_DIR}" /data/hf

if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  huggingface-cli login --token "${HUGGING_FACE_HUB_TOKEN}" --add-to-git-credential || true
fi

TARGET_MODEL_DIR="${BITNET_MODEL_DIR}/${BITNET_MODEL_NAME}"
MODEL_PATH="${TARGET_MODEL_DIR}/ggml-model-${BITNET_QUANT_TYPE}.gguf"
SERVER_PATH="build/bin/llama-server"

if [[ ! -f "${MODEL_PATH}" ]]; then
  mkdir -p "${TARGET_MODEL_DIR}"
  if command -v hf >/dev/null 2>&1; then
    hf download "${BITNET_GGUF_REPO}" "ggml-model-${BITNET_QUANT_TYPE}.gguf" --local-dir "${TARGET_MODEL_DIR}"
  else
    huggingface-cli download "${BITNET_GGUF_REPO}" "ggml-model-${BITNET_QUANT_TYPE}.gguf" --local-dir "${TARGET_MODEL_DIR}"
  fi
fi

if [[ ! -x "${SERVER_PATH}" || ! -f "${MODEL_PATH}" ]]; then
  echo "Preparing BitNet runtime and model metadata..."

  setup_base=(
    --model-dir "${TARGET_MODEL_DIR}"
    --log-dir "${BITNET_LOG_DIR}"
    --quant-type "${BITNET_QUANT_TYPE}"
  )

  if [[ "${BITNET_USE_PRETUNED}" == "1" ]]; then
    if ! python3 setup_env.py "${setup_base[@]}" --use-pretuned; then
      echo "Pretuned setup failed; retrying without pretuned kernels..."
      python3 setup_env.py "${setup_base[@]}"
    fi
  else
    python3 setup_env.py "${setup_base[@]}"
  fi
fi

# NOTE: llama-server does not expose `-cnv`; for OpenAI `/v1/chat/completions`
# the server uses model metadata or an explicit chat template when provided.
server_cmd=(
  "${SERVER_PATH}"
  -m "${MODEL_PATH}"
  -c "${BITNET_CTX_SIZE}"
  -t "${BITNET_THREADS}"
  -n "${BITNET_N_PREDICT}"
  -ngl "0"
  --temp "${BITNET_TEMPERATURE}"
  --host "${BITNET_HOST}"
  --port "${BITNET_PORT}"
)

if [[ "${BITNET_ENABLE_CONT_BATCH}" == "1" ]]; then
  server_cmd+=(-cb)
fi

if [[ -n "${BITNET_CHAT_TEMPLATE}" ]]; then
  server_cmd+=(--chat-template "${BITNET_CHAT_TEMPLATE}")
fi

if [[ -n "${BITNET_OVERRIDE_KV}" ]]; then
  server_cmd+=(--override-kv "${BITNET_OVERRIDE_KV}")
fi

exec "${server_cmd[@]}"
