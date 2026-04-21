#!/bin/sh -l

#SBATCH --job-name=synthclinclit-extract
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -euo pipefail
mkdir -p logs

# ---- Paths (EDIT) ----------------------------------------------------------
PROJECT_ROOT="/scratch/users/balshaikhdeeb/omop_nlp_project"
OLLAMA_SIF="${OLLAMA_SIF:-$PROJECT_ROOT/containers/ollama.sif}"   # Apptainer image
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$PROJECT_ROOT/models}"   # persistent model cache
PROJECT_DIR="${PROJECT_DIR:-$HOME/synth-clinic-lit}"
SCHEMA="${SCHEMA:-examples/schema_clinical_trial.json}"
# Input source — set EITHER IDS_FILE (resolve + download + extract)
# OR XML_DIR (skip download, extract from local JATS XML files).
IDS_FILE="${IDS_FILE:-}"            # e.g. ids.txt
XML_DIR="${XML_DIR:-}"              # e.g. ./articles
RESULTS="${RESULTS:-extractions.json}"
MODEL="${MODEL:-llama3.1:8b}"
NUM_CTX="${NUM_CTX:-16384}"

if [[ -z "${IDS_FILE}" && -z "${XML_DIR}" ]]; then
    echo "ERROR: set IDS_FILE or XML_DIR" >&2
    exit 2
fi
if [[ -n "${IDS_FILE}" && -n "${XML_DIR}" ]]; then
    echo "ERROR: set only one of IDS_FILE / XML_DIR, not both" >&2
    exit 2
fi

# ---- Ollama server config --------------------------------------------------
# Pick a free-ish port (stagger by job id so parallel jobs don't clash).
export OLLAMA_PORT=$((11434 + SLURM_JOB_ID % 1000))
# Server-side env: no scheme (Ollama convention).
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
# Client URL: synthclinclit's --ollama-host expects a full http:// URL.
OLLAMA_URL="http://127.0.0.1:${OLLAMA_PORT}"
mkdir -p "$OLLAMA_MODELS_DIR"

module load env/release/2023b
module load tools/Apptainer/1.4.0
#module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

cd "$PROJECT_DIR"
source .venv/bin/activate

# ---- Start Ollama server in background -------------------------------------
echo "[$(date)] Starting Ollama server on ${OLLAMA_HOST}"
apptainer exec --nv \
    --env OLLAMA_HOST="${OLLAMA_HOST}" \
    --env OLLAMA_MODELS=/root/.ollama/models \
    --bind "${OLLAMA_MODELS_DIR}:/root/.ollama" \
    "${OLLAMA_SIF}" ollama serve &
OLLAMA_PID=$!
trap 'echo "[$(date)] stopping ollama ($OLLAMA_PID)"; kill $OLLAMA_PID 2>/dev/null || true' EXIT

# ---- Wait for the server to be ready ---------------------------------------
READY=0
for i in $(seq 1 120); do
    if curl -sf "${OLLAMA_URL}/api/tags" >/dev/null; then
        echo "[$(date)] Ollama ready after ${i}s"
        READY=1
        break
    fi
    sleep 1
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "Ollama server died before becoming ready" >&2
        exit 1
    fi
done
if [[ "$READY" != "1" ]]; then
    echo "ERROR: Ollama did not become ready at ${OLLAMA_URL}" >&2
    exit 1
fi

# ---- Pull the model (no-op if already cached) ------------------------------
apptainer exec --nv \
    --env OLLAMA_HOST="${OLLAMA_HOST}" \
    --bind "${OLLAMA_MODELS_DIR}:/root/.ollama" \
    "${OLLAMA_SIF}" ollama pull "${MODEL}"

# ---- Run the extraction ----------------------------------------------------
# The CLI ignores OLLAMA_HOST — it takes --ollama-host explicitly.
SOURCE_ARGS=()
if [[ -n "${XML_DIR}" ]]; then
    SOURCE_ARGS+=(--xml-dir "${XML_DIR}")
else
    SOURCE_ARGS+=(-f "${IDS_FILE}")
fi

synthclinclit extract \
    -s "${SCHEMA}" \
    "${SOURCE_ARGS[@]}" \
    -r "${RESULTS}" \
    --backend ollama \
    --ollama-host "${OLLAMA_URL}" \
    --model "${MODEL}" \
    --num-ctx "${NUM_CTX}"

echo "[$(date)] Done"
