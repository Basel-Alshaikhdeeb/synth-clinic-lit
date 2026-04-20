#!/bin/bash
#SBATCH --job-name=synthclinclit-extract
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu          # EDIT: your GPU partition
##SBATCH --account=your_account  # EDIT: uncomment + set if required

set -euo pipefail
mkdir -p logs

# ---- Paths (EDIT) ----------------------------------------------------------
OLLAMA_SIF="${OLLAMA_SIF:-$HOME/containers/ollama.sif}"   # Apptainer image
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$HOME/.ollama}"   # persistent model cache
PROJECT_DIR="${PROJECT_DIR:-$HOME/synth-clinic-lit}"
SCHEMA="${SCHEMA:-examples/schema_clinical_trial.json}"
IDS_FILE="${IDS_FILE:-ids.txt}"
MODEL="${MODEL:-llama3.1:8b}"
NUM_CTX="${NUM_CTX:-16384}"

# ---- Ollama server config --------------------------------------------------
# Pick a free-ish port (stagger by job id so parallel jobs don't clash).
export OLLAMA_PORT=$((11434 + SLURM_JOB_ID % 1000))
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
mkdir -p "$OLLAMA_MODELS_DIR"

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

cd "$PROJECT_DIR"

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
for i in $(seq 1 60); do
    if curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null; then
        echo "[$(date)] Ollama ready after ${i}s"
        break
    fi
    sleep 1
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "Ollama server died before becoming ready" >&2
        exit 1
    fi
done

# ---- Pull the model (no-op if already cached) ------------------------------
apptainer exec --nv \
    --env OLLAMA_HOST="${OLLAMA_HOST}" \
    --bind "${OLLAMA_MODELS_DIR}:/root/.ollama" \
    "${OLLAMA_SIF}" ollama pull "${MODEL}"

# ---- Run the extraction ----------------------------------------------------
# synthclinclit reads OLLAMA_HOST automatically (ollama-python convention).
export OLLAMA_HOST

synthclinclit extract \
    -s "${SCHEMA}" \
    -f "${IDS_FILE}" \
    --backend ollama \
    --model "${MODEL}" \
    --num-ctx "${NUM_CTX}"

echo "[$(date)] Done"
