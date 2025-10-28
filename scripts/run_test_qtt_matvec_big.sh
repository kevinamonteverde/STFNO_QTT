#!/usr/bin/env bash
# Wrapper for running the big QTT/TT matvec tests
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
module load python 2>/dev/null || true
cd "$REPO_ROOT" || exit 1
python3 scripts/test_qtt_matvec_big.py "$@"
