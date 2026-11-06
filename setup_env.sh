#!/usr/bin/env bash
# Recreates the .venv for SemCoupling from scratch (macOS / Linux).
# Usage (from the SemCoupling folder):
#   bash setup_env.sh          # core only
#   bash setup_env.sh --nlp    # core + NLP stack (~2-3 GB)
set -euo pipefail
cd "$(dirname "$0")"

NLP=0
[[ "${1:-}" == "--nlp" ]] && NLP=1

# Pick a Python >= 3.11 (pandas 3 needs it)
PY=""
for cand in python3.11 python3.12 python3.13 python3; do
    if command -v "$cand" >/dev/null 2>&1 && \
       "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
        PY="$cand"; break
    fi
done
if [[ -z "$PY" ]]; then
    echo "ERROR: Python >= 3.11 not found. Install it (macOS: 'brew install python@3.11', Ubuntu: 'sudo apt install python3.11 python3.11-venv')."
    exit 1
fi
echo "Using $($PY --version) ($PY)"

if [[ -d .venv ]]; then
    echo "Removing old .venv ..."
    rm -rf .venv
fi
"$PY" -m venv .venv
VPY=".venv/bin/python"

"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -r requirements.txt
if [[ $NLP -eq 1 ]]; then
    "$VPY" -m pip install -r requirements-nlp.txt
    "$VPY" -m nltk.downloader punkt punkt_tab      # Polish tokenizer models for NLTK
fi

# Register the kernel so VS Code / Jupyter shows it by name
"$VPY" -m ipykernel install --user --name semcoupling --display-name "Python (SemCoupling)"

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "In VS Code pick kernel: 'Python (SemCoupling)'"
