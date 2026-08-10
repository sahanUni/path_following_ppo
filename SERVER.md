# Running batches on the compute server

The laptop stays the interactive machine: dashboard, MuJoCo replay, analysis.
The server is for batch training only. Nothing here needs a display.

## 1. Get the code across

```bash
git clone <repo-url> Path_Following_PPO
cd Path_Following_PPO
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `numpy`, `mujoco`, `gymnasium` and `stable-baselines3`
exactly. Keep it that way: MuJoCo is the physics, and a different patch version
can move trajectories enough to break the parity check below.

The clone carries `runs/calibration_v7.json` (the baseline gains and the action
box -- an experiment INPUT, not an output), plus `runs/v7_seed7/final_model.zip`
and `runs/v7_seed7/confirm_200.csv` as the known-good reference. Everything else
under `runs/` is ignored.

## 2. Check the machine agrees with the laptop, before spending compute on it

Two levels. Do the first one; it is thirty minutes and it has caught real
problems.

**Deterministic check.** No training, so no seed noise -- the same scenarios
must give the same answers:

```bash
python -m unittest discover -s tests -t .
python confirm_advantage.py \
  --model runs/v7_seed7/final_model.zip \
  --calibration runs/calibration_v7.json \
  --episodes 200 --extra-kp 15,20,34,42,50 \
  --output runs/v7_seed7/confirm_200_server.csv
```

Compare against `runs/v7_seed7/confirm_200.csv` (laptop):

| controller | completed |
|---|---|
| PPO | 173/200 |
| fixed Kp 15 | 159/200 |
| fixed Kp 20 | 159/200 |
| fixed Kp 26 (baseline) | 158/200 |
| fixed Kp 34 | 151/200 |
| fixed Kp 42 | 143/200 |
| fixed Kp 50 | 140/200 |

**The fixed-PID rows must match exactly.** They involve no neural network, so
any difference means the physics differs -- wrong MuJoCo version, or a different
calibration artifact. Stop and fix that before going further.

The PPO row may differ by an episode or two. Torch on a different CPU can order
floating-point reductions differently, which shifts an action in the last bits
and can flip a marginal episode. A difference of more than about three episodes
is not rounding and is worth investigating.

**Training check (optional).** A 100k-step run whose `monitor.csv` reward
deciles look like the laptop's. Do not expect identical numbers -- different
BLAS makes training trajectories diverge even from the same seed.

## 2b. This is a SLURM cluster

`sinfo -s` shows one partition, `base`, 48 nodes, no time limit. `login01` is
for editing and submitting only -- never for training. It is also tightly capped
on threads per user, which is what makes `uv` fall over with
`failed to initialize global rayon pool` unless `RAYON_NUM_THREADS` is small.

Two ways to run, both fine. Neither needs you to stay connected.

**Interactive, inside tmux.** The allocation lives on a compute node; tmux keeps
it alive across disconnects:

```bash
tmux new -s pf
srun --partition=base --cpus-per-task=6 --mem=16G --time=12:00:00 --pty bash
# now on a compute node
cd ~/path_following_ppo && source venv/bin/activate
python run_seeds.py --seeds 7,21,42,84,123 --total-timesteps 1000000 \
  --calibration runs/calibration_v7.json --run-root runs/rq1_blind
# Ctrl-b then d to detach; tmux attach -t pf to return
```

Good for watching the first run and catching a mistake early. The catch: if the
tmux session dies, the allocation and the runs die with it.

**Batch, fire and forget.** `slurm_batch.sh` runs the seeds and then the paired
confirmation for each:

```bash
sbatch slurm_batch.sh
squeue -u $USER
tail -f slurm-pf-rq1-<jobid>.out
```

Survives everything, including tmux dying. Override anything via `--export`:

```bash
sbatch --export=ALL,RUN_ROOT=runs/rq1_clean,EXTRA="--no-delay --no-noise" \
  --job-name=pf-clean slurm_batch.sh
```

### Core budget

`--cpus-per-task=6` for five seeds: one core each plus slack. Asking for a whole
node would not finish sooner -- the seeds already run concurrently, so the extra
cores would sit idle. Scale it with the seed count, not with what is available.

Note that `os.cpu_count()` inside a job reports the whole compute node, not the
allocation, so `run_seeds.py` uses `os.sched_getaffinity` instead and warns when
the requested threads exceed the granted cores.

## 3. Launch a batch

```bash
python run_seeds.py \
  --seeds 7,21,42,84,123 \
  --total-timesteps 1000000 \
  --calibration runs/calibration_v7.json \
  --run-root runs/rq1_blind
```

One process per seed, single-threaded each. `--dry-run` prints the commands
without running anything.

### Why one process per seed rather than parallel envs in one run

Measured on this project: env stepping is 47% of training wall time, the PPO
gradient update is the other 53%, and that half is serial. Env parallelism
therefore caps at roughly 1.7x however many cores it is given, while independent
seeds scale close to linearly -- five seeds on five cores beats one seed on five
cores by about 3x. It also leaves each run's training dynamics exactly what was
validated on one core, which matters when the batch is the reported result.

### Using 32 cores properly

Five seeds is five cores. The rest is not wasted if you queue the arms you
already have flags for -- these are independent batches and can run at once:

```bash
# the headline arm
python run_seeds.py --seeds 7,21,42,84,123 --calibration runs/calibration_v7.json \
  --run-root runs/rq1_blind &

# clean-plant ablation: the plant where raising Kp was free. The expected
# result is that PPO and the fixed baseline converge, which is the control
# showing the effect comes from the plant having a real tradeoff.
python run_seeds.py --seeds 7,21,42,84,123 --calibration runs/calibration_v7.json \
  --run-root runs/rq1_clean --extra="--no-delay --no-noise" &

# one imperfection at a time, to attribute the effect
python run_seeds.py --seeds 7,21,42 --calibration runs/calibration_v7.json \
  --run-root runs/rq1_nodelay --extra="--no-delay" &
python run_seeds.py --seeds 7,21,42 --calibration runs/calibration_v7.json \
  --run-root runs/rq1_nonoise --extra="--no-noise" &
```

That is 21 cores on four arms that all feed the write-up. Every batch drops a
`batch_manifest.json` recording seeds, artifact, flags, and machine.

### The observation arms (RQ2)

Kept separate so the effect is attributable. `preview` is TASK information and
is not privileged -- a deployed robot knows its own planned path. `plant
context` is genuinely privileged: a real robot does not know its own dead time.
Blurring them would let a reviewer object that the "privileged" agent mostly won
on information that was not privileged.

```bash
python run_seeds.py --seeds 7,21,42,84,123 --calibration runs/calibration_v7.json \
  --run-root runs/rq2_preview --extra="--preview" &
python run_seeds.py --seeds 7,21,42,84,123 --calibration runs/calibration_v7.json \
  --run-root runs/rq2_context --extra="--plant-context" &
python run_seeds.py --seeds 7,21,42,84,123 --calibration runs/calibration_v7.json \
  --run-root runs/rq2_both --extra="--preview --plant-context" &
```

Each arm changes the observation width, so its models are not interchangeable
with the blind ones. `train.py` writes `arms.json` beside each model and
`confirm_advantage.py` reads it, because the width alone is ambiguous -- preview
and plant context add five values each, so 175 could be either. A mismatch is
caught and reported rather than surfacing as a shape error inside SB3.

The prediction worth checking first: blind agents show
`corr(mean episode Kp, dead time) = +0.08`, i.e. no plant identification at all.
The context arm should push that clearly negative.

Only reach for `SubprocVecEnv` when a SINGLE run must be fast -- the
direct-control agent, most likely, where the budget is large and there is only
one of it. Three things need fixing first, none of which matter today:
`n_steps` is per environment (so `n_envs=8` gives 8x fewer gradient updates
unless `n_steps` is divided to match); `train.py` gives every env the same
`monitor.csv` filename, which multiple workers would corrupt; and
`CurriculumAndMetricsCallback` calls `env_method("set_training_progress", ...)`
every step, which becomes an IPC round trip per worker per step and would eat
most of the speedup.

## 4. Threads

`--threads-per-job` defaults to 1, and the launcher also exports
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`NUMEXPR_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` to match. Without this every
process grabs every core: five runs each spawning 32 threads on 32 cores is
slower than five single-threaded ones.

Raise it only when the batch is smaller than the core count and the spare cores
would otherwise idle. Thread count changes floating-point reduction order, so
keep it fixed across runs that are meant to be compared.

## 5. Collect the results

```bash
for seed in 7 21 42 84 123; do
  python confirm_advantage.py \
    --model runs/rq1_blind/seed$seed/best_model.zip \
    --calibration runs/calibration_v7.json \
    --episodes 200 --extra-kp 15,20,34,42,50 \
    --output runs/rq1_blind/seed$seed/confirm_200.csv
done
```

Copy the `confirm_200.csv` files and `evaluation_history.csv` back to the laptop
for plotting. The model zips are ~400 KB each if you want them for the
dashboard; the checkpoints and tensorboard directories are the bulk and are
rarely worth moving.

## Troubleshooting

- **`invalid value for environment variable MUJOCO_GL`** -- something set it to
  a backend this machine lacks. The launcher deliberately leaves it unset,
  because training never renders and mujoco only needs a backend when a
  rendering context is actually created. If a headless box does complain,
  `export MUJOCO_GL=osmesa`.
- **Runs die instantly** -- read `runs/<batch>/seed<N>/train.log`; the launcher
  captures stdout and stderr there and reports a non-zero exit per seed.
- **Everything is slower than the laptop per run** -- check nothing else is
  sharing the cores, and that `--threads-per-job` is not fighting the batch
  size.
