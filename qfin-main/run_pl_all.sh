#!/usr/bin/env bash
# PennyLane 版：一次跑完兩個實驗（在 Linux／WSL2 的 Ubuntu 執行）。
# 用法：bash run_pl_all.sh [CPU 核心數，預設 16]
set -euo pipefail
CORES="${1:-16}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")"
mkdir -p logs

echo "== 0. 檢查 PennyLane 電路和 numpy 模擬器是否一致"
python pl_check.py | tee logs/pl_check.log

echo "== 1. 實驗一：QLSTM vs QFWP（10 個種子 × 6 種 batch）"
python pl_exp1.py --seeds 10 --jobs "$CORES" --lr 0.05 --opt adam | tee logs/pl_exp1.log
python report1.py pl_adam0.05 | tee logs/pl_exp1_report.txt

echo "== 2. 實驗二：QLSTM 預測器（3 個種子同時跑）"
for s in 0 1 2; do python pl_exp2.py qlstm --seeds 1 --seed0 $s > logs/pl_qlstm_s$s.log 2>&1 & done
wait
grep -h "test acc" logs/pl_qlstm_s*.log | grep seed

echo "== 3. 實驗二：A3C 與 QA3C（每個種子 8 個 worker；同時跑 CORES/8 個種子）"
PAR=$(( CORES / 8 )); [ "$PAR" -lt 1 ] && PAR=1
run_agent () {
  local agent=$1 n=$2 i=0
  for s in $(seq 0 $((n-1))); do
    python pl_exp2.py rl --agent "$agent" --seeds 1 --seed0 "$s" > "logs/pl_${agent}_s$s.log" 2>&1 &
    i=$((i+1)); if [ "$i" -ge "$PAR" ]; then wait; i=0; fi
  done
  wait
}
run_agent A3C 3
run_agent QA3C 3
grep -h "測試" logs/pl_A3C_s*.log logs/pl_QA3C_s*.log
python report2.py exp2_pl | tee logs/pl_exp2_report.txt
echo "完成。結果在 results/exp1a_pl_adam0.05/、results/exp2_pl/，紀錄在 logs/"
