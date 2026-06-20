#!/usr/bin/env bash
# Run the lidar-align test suite against the project venv.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" GLOG_minloglevel=3
PY=./.venv/Scripts/python.exe
[ -x "$PY" ] || PY=python          # fall back to PATH python on Linux/WSL

fail=0
for t in test_index test_prealign test_global test_synth test_outofcore test_export test_scale; do
  echo "=== $t ==="
  if "$PY" "tests/$t.py" 2>&1 | grep -E "PASS|FAIL|Error|assert"; then :; fi
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "  -> FAILED"; fail=1; }
done
exit $fail
