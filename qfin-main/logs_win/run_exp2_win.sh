#!/usr/bin/env bash
# Windows rerun of Exp 2 (results -> results_win/exp2_pl). Seeds run concurrently (8 workers each).
set -u
cd "$(dirname "$0")/.."
export QFIN_RES="$(pwd)/results_win" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8
PY=./.venv-pl/Scripts/python.exe
echo "== forecaster $(date)"
for s in 0 1 2; do $PY pl_exp2.py qlstm --seeds 1 --seed0 $s > logs_win/pl_qlstm_s$s.log 2>&1 & done; wait
grep -h "test acc" logs_win/pl_qlstm_s*.log | grep seed
echo "== A3C $(date)"
for s in 0 1 2; do $PY pl_exp2.py rl --agent A3C --seeds 1 --seed0 $s > logs_win/pl_A3C_s$s.log 2>&1 & done; wait
grep -h "測試" logs_win/pl_A3C_s*.log
echo "== QA3C $(date)"
for s in 0 1 2; do $PY pl_exp2.py rl --agent QA3C --seeds 1 --seed0 $s > logs_win/pl_QA3C_s$s.log 2>&1 & done; wait
grep -h "測試" logs_win/pl_QA3C_s*.log
echo "== report $(date)"
$PY report2.py exp2_pl > logs_win/pl_exp2_report.txt 2>&1
echo "== done $(date)"
