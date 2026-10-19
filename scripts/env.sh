# Source this before any dsd-real run.
export DSD_VENV=${DSD_VENV:-/scratch/yf3005/miniconda3/envs/vllm027}   # override with your own venv
export PATH="$DSD_VENV/bin:$PATH"          # flashinfer JIT shells out to bare `ninja`
export PY="$DSD_VENV/bin/python"
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-ERROR}
export TOKENIZERS_PARALLELISM=false
