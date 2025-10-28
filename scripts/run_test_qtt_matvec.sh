#!/usr/bin/env bash
# Wrapper to ensure the Python module is loaded in module environments (e.g., Lmod/Tcl modules)
# and then run the QTT contraction test script.
# Usage: ./scripts/run_test_qtt_matvec.sh

# Determine script and repository root directories and export them to PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Try to load the system Python module; harmless if not available in this shell.
module load python 2>/dev/null || true

# Run the python test script from the repository root (ensure we run from repo root)
cd "$REPO_ROOT" || exit 1
python3 scripts/test_qtt_matvec.py "$@"
