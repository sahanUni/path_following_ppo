# Path Following PPO

A self-contained development project in which a blind Stable-Baselines3 PPO
policy tunes only the steering PID gains `(Kp, Ki, Kd)` online. The speed PID,
MuJoCo plant, path projection, and steering-priority wheel allocator are shared
by the learned controller and the calibrated fixed-PID baseline.

## Controller contract

- MuJoCo and both PIDs run at 500 Hz (`dt = 0.002 s`).
- PPO runs at 50 Hz and selects three bounded absolute steering-gain targets.
- The speed target is fixed at 0.3, 0.5, or 0.7 m/s; achieved speed may fall in
  corners because steering receives first claim on wheel authority.
- The policy gets ten causal observation frames (0.2 s), with no mass,
  friction, actuator strength, path identity, progress, curvature, or preview.
- True point-to-path distance is used by the reward and evaluation, but is not
  an observation supplied to the deployed policy.
- The vehicle always starts at `(0, 0, 0.03)`, aligned with `+x`, exactly as in
  `Path_Following_Sandbox`.

See [DEVELOPMENT_SPEC.md](DEVELOPMENT_SPEC.md) for the frozen design.

## Plant imperfections: actuator dead time and sensor noise

Added 2026-08-09, because without them the study had no result to report.

On the original plant the car read its cross-track error exactly and its wheel
command landed the same instant it was computed. Under those conditions raising
`Kp` is monotonically better until the wheel clamp truncates the command, so the
best fixed gain is simply whichever one saturates first. PPO correctly pinned
its `Kp` action at the box ceiling, the Fixed PID baseline sat at the same
place, and the two produced the same trajectories. There was no tradeoff to
schedule, and `Kd` was very nearly a decorative action channel.

Two imperfections restore the tradeoff, both drawn per episode from
`config.ACTUATOR_DELAY_RANGE_S` and `config.SENSOR_NOISE_RANGE_M`:

- **Dead time** queues the wheel command for a whole number of physics steps, so
  a correction lands after the error it was computed for has moved on. The
  overshoot grows with `Kp`, which puts a stability ceiling on it. At 100 ms the
  ranking on `arc` inverts outright: 0.98 cm at `Kp` 20 against 6.78 cm at 50.
- **Sensor noise** corrupts only the `e_ct` the steering PID measures. It acts
  almost entirely through `Kd`, because the derivative of a clean signal is free
  information and the derivative of a noisy one is mostly amplified garbage.

They pull `Kd` in opposite directions -- dead time makes damping mandatory
(`Kd = 0` fails at 100 ms), noise makes it expensive -- so both gains land on an
interior optimum, and that optimum **moves with the condition**. At the
checkpoint-selection operating point the best corner of the gain box is
`(Kp low, Kd low)` on `arc` at 0.3 m/s and `(Kp high, Kd high)` on `zigzag` at
0.7 m/s. No single fixed gain is right in both places, which is the precondition
the whole gain-scheduling question depends on.

Two rules the implementation keeps:

- **Two error signals.** Reward, metrics, and corridor termination use the true
  state; only the PID and the observation see the noisy one. Scoring the
  controller on its own noisy measurement would report performance that did not
  happen.
- **Neither value appears in the observation.** They are hidden plant parameters
  in the same sense mass and friction already are: the blind student must infer
  them from the error history, the privileged teacher can be told.

`train.py --no-delay` and `--no-noise` are the ablation controls, and each
removes only its own imperfection without shifting the other's random draws.
`GainTradeoffTests` in `tests/test_environment.py` asserts the tradeoff exists,
including a test that still asserts the OLD behaviour on a clean plant, so the
reasoning behind the ranges cannot silently rot.

## Setup

From `D:\Msc\Experiments\Path_Following_PPO`:

```powershell
..\venv\Scripts\python.exe -m pip install -r requirements.txt
..\venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 1. Calibrate the fixed PID first

The training command intentionally requires a calibration artifact. Calibration
ranks gain sets by completion first, mean true distance second, ITAE third, and
gain magnitude last. It then saves the global baseline and the PPO gain box.

Development-sized calibration:

```powershell
..\venv\Scripts\python.exe calibrate_pid.py `
  --candidates 48 `
  --physics-samples 2 `
  --output runs\calibration.json
```

Fast pipeline check (not a defensible baseline):

```powershell
..\venv\Scripts\python.exe calibrate_pid.py `
  --paths arc `
  --speeds 0.3 `
  --candidates 2 `
  --physics-samples 0 `
  --output runs\smoke_calibration.json
```

## 2. Train PPO

First diagnostic run:

```powershell
..\venv\Scripts\python.exe train.py `
  --calibration runs\calibration.json `
  --total-timesteps 100000 `
  --run-dir runs\development
```

Only continue to one million steps after the 100k logs show improving
completion and distance:

```powershell
..\venv\Scripts\python.exe train.py `
  --calibration runs\calibration.json `
  --total-timesteps 1000000 `
  --run-dir runs\development_1m
```

Training writes TensorBoard data, per-episode CSV metrics, checkpoints every
25k steps, a deterministic evaluation history, `best_model.zip`, and
`final_model.zip`.

View TensorBoard:

```powershell
..\venv\Scripts\python.exe -m tensorboard.main --logdir runs\development\tensorboard
```

## 3. Evaluate identically

Fixed PID:

```powershell
..\venv\Scripts\python.exe evaluate.py `
  --fixed-pid `
  --calibration runs\calibration.json `
  --path-set training `
  --plot
```

PPO:

```powershell
..\venv\Scripts\python.exe evaluate.py `
  --model runs\development\best_model.zip `
  --calibration runs\calibration.json `
  --path-set training `
  --plot
```

Change `--path-set` to `held-out`, `stress`, or `all`. The evaluator saves one
summary CSV, one detailed 50 Hz trace CSV, and optional episode PNGs showing
trajectory, error, gains, and wheel commands.

## 4. Run the paired adaptation evaluator

Start with the four-scenario smoke matrix. It covers nominal physics, one mass
step, and lateral force pulses in both directions:

```powershell
..\venv\Scripts\python.exe adaptation_evaluate.py `
  --model runs\development\best_model.zip `
  --calibration runs\calibration.json `
  --matrix smoke
```

The full one-seed development matrix uses seven representative paths, all
three target speeds, stationary one-axis-at-a-time physics variations, two
mass steps, and four force pulses:

```powershell
..\venv\Scripts\python.exe adaptation_evaluate.py `
  --model runs\development\best_model.zip `
  --calibration runs\calibration.json `
  --matrix development
```

Use `--paths` and `--speeds` to create an inspectable subset before running
the full matrix. Each invocation creates a new directory under
`runs/adaptation_evaluation` containing:

```text
manifest.json
episode_summary.csv
paired_summary.csv
aggregate.json
traces/<scenario_id>__fixed.csv
traces/<scenario_id>__ppo.csv
logs/
plots/
```

The manifest records the fully expanded scenarios, event timing, recovery
configuration, package versions, source hashes, calibration bounds, and
SHA-256 hashes for the PPO and calibration artifacts. A run ID cannot overwrite
an existing run.

## 5. Inspect results in the dashboard

```powershell
..\venv\Scripts\python.exe dashboard.py
```

Open `http://127.0.0.1:8060`. Select a path and target speed. The dashboard runs
both controllers live on the same episode -- it reads no stored run history, so
what you see is always the current code and the current sliders. Episodes are
deterministic and cached in memory for the lifetime of the process.

Left panel is the **Fixed PID with hand-editable gains**. `Kp`, `Ki`, and `Kd`
default to the calibrated baseline and re-simulate the moment you move them, so
you can hunt for a better gain by hand and watch the trajectory redraw. The
slider bounds are the calibration SEARCH space, not the narrower PPO action box,
deliberately: part of the value is trying gains the policy cannot reach and
seeing where the box should have been. "Reset to calibrated baseline" restores
the artifact's values.

Right panel is the **trained PPO model**, on the identical scenario. Its
subtitle reports the mean and the min-max range of each gain the policy actually
applied. The range is the part to read: a scheduler that settles on one constant
is the null result this project keeps rediscovering, and a mean alone hides it.

Moving a gain slider re-runs only the Fixed PID -- the PPO episode does not
depend on those gains and is cached against the scenario alone.

The **plant imperfection** sliders (dead time in ms, sensor noise in mm) are the
ones to reach for first. Set both to zero and the two panels converge on the
same answer with `Kp` at the ceiling, which is the degenerate case the study
started from. Raise dead time past ~60 ms and the ranking inverts in front of
you.

The dashboard also runs **before any model exists**: if
`runs\development\best_model.zip` is absent, the left panel works normally and
the right shows the command to train one. Point it elsewhere with
`--model` and `--calibration`; a model and the calibration it was trained
against must move together, since the action vector is interpreted through
`low`/`base`/`high`. Without `--calibration` it picks the newest of
`calibration_v6.json`, `calibration_v5.json`, `calibration_twosided.json`.

```powershell
..\venv\Scripts\python.exe dashboard.py `
  --calibration runs/calibration_v6.json `
  --model runs/development/best_model.zip `
  --port 8060
```

### Charts in each panel

- **trajectory** -- corridor, reference path, driven line, and the wind drawn as
  arrows along the track. Direction is the half of a disturbance the tracking
  error responds to: a gust that veers across the path costs far more than one
  of the same size blowing along it, and magnitude alone cannot show that.
- **tracking error** -- `e_ct` against true distance off path.
- **speed** -- achieved against target, with allocation-limited samples shaded.
  Steering has first claim on the wheels, so the speed channel is what gives way
  in a corner; the car brakes itself into apexes without anything asking it to.
- **external force** -- total magnitude plus world-frame components.
- **steering gains** -- what the controller did with `Kp`/`Ki`/`Kd` against the
  box it is allowed to use. A flat line here is the null result to watch for.

### Watch a run in MuJoCo

"Replay Fixed PID" and "Replay PPO" in the drawer open `replay.py` in a separate
window with the current settings, wind drawn as an arrow over the car. It runs
as its own process because MuJoCo's viewer wants the main thread and would fight
the Dash server. Standalone:

```powershell
..\venv\Scripts\python.exe replay.py --controller ppo `
  --path zigzag --speed 0.7 --gust 8 --delay-ms 60 --noise-mm 0.3 --loop
```

The episode is re-simulated rather than shipped over as data, which is only safe
because the environment is deterministic given the same options and seeds. That
determinism is asserted by `test_the_replayed_episode_is_the_dashboard_episode`
rather than assumed, and the metrics are printed on startup so you can check
them against the panel.

## 6. Confirm the advantage is real

```powershell
..\venv\Scripts\python.exe confirm_advantage.py `
  --model runs/v7_seed7/final_model.zip `
  --calibration runs/calibration_v7.json `
  --episodes 200 --extra-kp 15,20,34,42,50 `
  --output runs/v7_seed7/confirm_200.csv
```

Answers one question with a defensible number: does the policy beat the fixed
baseline, or did it draw an easier set of episodes? Scenarios are **sampled from
the stage-3 training distribution**, not hand-picked -- readable conditions are
readable precisely because everything completes on them, and on those the gain
moves tracking by a fraction of a millimetre while the axis it actually controls
is whether the episode finishes at all.

Every controller replays the identical scenario, so the comparison is paired and
scenario difficulty cancels. Completion is tested with an exact McNemar
statistic over the episodes where the two controllers **disagree** -- the ones
both finish, or both fail, carry no information about which is better. At 60
episodes the random wobble in a completion count is around +/-5 episodes, which
swamps the difference being measured; 200 brings it to about +/-2.5.

With two or more fixed references it also reports the **per-scenario fixed-gain
oracle**: allow a different fixed gain for every scenario, chosen with hindsight.
Beating that cannot be done by any fixed gain, only by varying within the
episode.

## 7. Compare the observation arms (RQ2)

`summarise_batch.py` asks whether one batch beats the fixed PID. `compare_arms.py`
asks the RQ2 question instead: how much does information the blind agent does not
have actually buy, and does it buy a **scheduler** or just a better constant.

```powershell
..\venv\Scripts\python.exe compare_arms.py `
  --arm blind=runs/rq1_blind `
  --arm preview=runs/rq2_preview `
  --arm context=runs/rq2_context `
  --arm both=runs/rq2_both
```

The first arm is the reference the others are tested against. Nothing needs
re-running: `confirm_advantage.py` draws its scenarios policy-free from a fixed
`--scenario-seed`, so every arm has already replayed the identical episodes and
the comparison is paired scenario by scenario.

Three sections come out:

- **integrity** -- the fixed-gain baseline involves no network, so its completion
  count must be identical across arms. A mismatch means the arms ran different
  scenario suites, and the comparison would measure the suite rather than the
  observation; it aborts rather than reporting a number.
- **paired completion** -- exact McNemar per seed over the scenarios where the two
  arms disagree, then a sign test across seeds. Tracking is read only on episodes
  both arms finished.
- **scheduling** -- `corr(mean episode Kp, hidden parameter)` per arm. This is the
  headline. The blind agent sits near zero on dead time across all five seeds: it
  found a constant and reacts within the episode. A privileged arm that has
  genuinely identified the plant must push the delay column clearly **negative**,
  in every seed, not just win on completion. Winning without moving that number
  means the information reached the observation and never reached the behaviour.

Arms must cover the same seeds; `--allow-partial-seeds` compares the shared ones
mid-batch, with a warning that belongs in any table built that way.

## Path split

- Training: `arc`, `scurve`, `uturn`, `slalom`, `zigzag`, `spiral`
- Held out: `figure8`, `hairpin`, `square`, `zigzag60`
- Stress only: `zigzag30`, `needle`

## Development success gate

With the one development seed, PPO should at least match the calibrated fixed
PID's completion rate, reduce aggregate path distance on feasible scenarios,
and show nonconstant gain behavior when the plant changes. This is a pipeline
and feasibility result, not yet a statistical claim. Multi-seed evaluation
remains required before making statistical claims.
