#!/bin/bash
# Multi-seed training as a SLURM batch job.
#
#   sbatch slurm_batch.sh
#   sbatch --export=ALL,RUN_ROOT=runs/rq1_clean,EXTRA="--no-delay --no-noise" slurm_batch.sh
#
# One process per seed, one thread each. Env stepping is 47% of training wall
# time and the PPO gradient update is the other 53% and serial, so parallel
# environments inside one run cap at roughly 1.7x, while independent seeds
# scale nearly linearly. Five seeds on five cores beats one seed on five.
#
# Deliberately modest: 6 cores of a 48-node partition, not a whole node. The
# batch finishes in the same wall time either way -- the seeds run concurrently
# regardless -- so grabbing more would only idle them.

#SBATCH --job-name=pf-rq1
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SEEDS="${SEEDS:-7,21,42,84,123}"
TIMESTEPS="${TIMESTEPS:-1000000}"
CALIBRATION="${CALIBRATION:-runs/calibration_v7.json}"
RUN_ROOT="${RUN_ROOT:-runs/rq1_blind}"
EXTRA="${EXTRA:-}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source venv/bin/activate

# run_seeds.py exports these to its children too; setting them here covers
# anything that reads them before the launcher starts.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none} on ${SLURM_JOB_PARTITION:-none}"
echo "cores     : ${SLURM_CPUS_PER_TASK:-unknown} allocated"
echo "python    : $(python -V 2>&1)"
echo "seeds     : ${SEEDS}"
echo "run root  : ${RUN_ROOT}"
echo "extra     : ${EXTRA:-<none>}"
echo

python run_seeds.py \
  --seeds "${SEEDS}" \
  --total-timesteps "${TIMESTEPS}" \
  --calibration "${CALIBRATION}" \
  --run-root "${RUN_ROOT}" \
  --threads-per-job 1 \
  --extra="${EXTRA}"

echo
echo "training done, running the paired confirmation for each seed"
for seed in ${SEEDS//,/ }; do
  model="${RUN_ROOT}/seed${seed}/best_model.zip"
  if [ -f "${model}" ]; then
    python confirm_advantage.py \
      --model "${model}" \
      --calibration "${CALIBRATION}" \
      --episodes 200 --extra-kp 15,20,34,42,50 \
      --output "${RUN_ROOT}/seed${seed}/confirm_200.csv"
  else
    echo "  seed ${seed}: no best_model.zip, skipping confirmation"
  fi
done
