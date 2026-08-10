#!/bin/bash
# Paired confirmation only, for a batch whose training has already finished.
#
#   sbatch slurm_confirm.sh
#   sbatch --export=ALL,RUN_ROOT=runs/rq1_clean slurm_confirm.sh
#
# slurm_batch.sh runs the confirmations in a serial loop after training, which
# is fine when it is the tail of a job that already took an hour. Run on its
# own it is the whole job, and 5 seeds x 7 controllers x 200 episodes in
# sequence is hours of one core while five sit idle. Each seed is independent,
# so they go concurrently here, one thread each -- same reasoning as
# run_seeds.py, and the same reason the thread caps are exported.

#SBATCH --job-name=pf-confirm
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL

set -uo pipefail

SEEDS="${SEEDS:-7,21,42,84,123}"
CALIBRATION="${CALIBRATION:-runs/calibration_v7.json}"
RUN_ROOT="${RUN_ROOT:-runs/rq1_blind}"
MODEL="${MODEL:-best_model.zip}"
EPISODES="${EPISODES:-200}"
EXTRA_KP="${EXTRA_KP:-15,20,34,42,50}"
OUTPUT="${OUTPUT:-confirm_${EPISODES}.csv}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source venv/bin/activate

# One thread per confirmation. Without this each of the five grabs the whole
# node and they contend. Thread count also changes floating-point reduction
# order, so keeping it at 1 keeps these comparable with the laptop reference.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none} on ${SLURM_JOB_PARTITION:-none}"
echo "cores     : ${SLURM_CPUS_PER_TASK:-unknown} allocated"
echo "run root  : ${RUN_ROOT}"
echo "model     : ${MODEL}"
echo "episodes  : ${EPISODES}   extra Kp: ${EXTRA_KP}"
echo

pids=()
seeds=()
for seed in ${SEEDS//,/ }; do
  model="${RUN_ROOT}/seed${seed}/${MODEL}"
  if [ ! -f "${model}" ]; then
    echo "  seed ${seed}: no ${model}, skipping"
    continue
  fi
  log="${RUN_ROOT}/seed${seed}/confirm.log"
  echo "  seed ${seed}: ${model} -> ${RUN_ROOT}/seed${seed}/${OUTPUT}  (log: ${log})"
  python confirm_advantage.py \
    --model "${model}" \
    --calibration "${CALIBRATION}" \
    --episodes "${EPISODES}" \
    --extra-kp "${EXTRA_KP}" \
    --output "${RUN_ROOT}/seed${seed}/${OUTPUT}" \
    > "${log}" 2>&1 &
  pids+=($!)
  seeds+=("${seed}")
done

if [ ${#pids[@]} -eq 0 ]; then
  echo "no models found under ${RUN_ROOT} -- nothing to confirm"
  exit 1
fi

echo
echo "${#pids[@]} confirmations running concurrently, waiting"

failed=()
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "  seed ${seeds[$index]}: ok"
  else
    echo "  seed ${seeds[$index]}: FAILED -- see ${RUN_ROOT}/seed${seeds[$index]}/confirm.log"
    failed+=("${seeds[$index]}")
  fi
done

echo
if [ ${#failed[@]} -gt 0 ]; then
  echo "seeds failed: ${failed[*]}"
  exit 1
fi
echo "all confirmations done"
for seed in ${SEEDS//,/ }; do
  csv="${RUN_ROOT}/seed${seed}/${OUTPUT}"
  [ -f "${csv}" ] && echo "  ${csv}  ($(wc -l < "${csv}") rows)"
done
