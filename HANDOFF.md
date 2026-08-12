# Handoff — Path_Following_PPO

Written 2026-08-12. Continues an MSc thesis experiment on online RL tuning of a
steering PID.

## Where everything already is

Do not re-derive any of this; it is written down.

| what | where |
|---|---|
| Code, all results tooling | `D:\Msc\Experiments\Path_Following_PPO` |
| Public repo | https://github.com/sahanUni/path_following_ppo |
| **Results and their reasoning, in plain language** | `FINDINGS.md` — read this first |
| Pipeline commands, dashboard, replay | `README.md` |
| Cluster setup, SLURM, batch launching | `SERVER.md` |
| Frozen design | `DEVELOPMENT_SPEC.md` |
| Paired adaptation evaluator spec | `PAIRED_ADAPTATION_EVALUATOR.md` |
| Persistent notes for future sessions | `C:\Users\sahan\.claude\projects\D--Msc-Experiments\memory\` |

Relevant memory files: `clean-plant-free-gain`, `completion-is-the-signal`,
`prize-is-within-episode`, `feedback-staged-compute`, `thesis-project-state`.

## State in one paragraph

The plant was fixed (it previously made high gain free, so the study could not
answer anything), a fair baseline was calibrated, and both research arms were
run at five seeds each on the cluster. The headline result is **negative but
well-evidenced**: per-episode gain selection is worth ~2% on this scenario
distribution, so RL matches a carefully calibrated fixed PID without beating it,
privileged plant context does not help, and no policy adapts its gain to
anything measurable. Full numbers and reasoning: `FINDINGS.md`.

## Decisions that are settled — do not relitigate

- **Plant imperfections stay.** Actuator dead time (0–80 ms) and cross-track
  sensor noise (0–0.4 mm) are what create a real gain trade-off. Ranges are
  measured, justified in `config.py` comments.
- **Two error signals.** The PID and the observation see noisy `e_ct`; reward,
  metrics and corridor termination use the true state. Never merge these.
- **`calibration_v7.json` is the baseline** (Kp 26, Ki 0.5, Kd 4.4; box
  15–50). The box was set by hand from measurements and is labelled
  `"box_source": "explicit"`. v6 is broken (picked a gain that "succeeds" by
  crawling) and is deliberately excluded from dashboard auto-resolution.
- **Preview and plant context are separate arms.** Preview is task information
  a deployed robot legitimately has; plant context is genuinely privileged.
  Merging them would make the effect unattributable.
- **Preview arm keeps 3 actions** (no speed action) — user decision.

## Traps that already cost time

- **Login node caps memory and threads.** Anything importing torch or MuJoCo —
  tests, confirmations, training — must run inside `srun`/`sbatch`, or it fails
  with `MemoryError` on a 700-byte file, or `uv` panics on its thread pool.
- **`srun` shells die with the terminal**, taking runs with them. Twice. Use
  `sbatch` for anything long.
- **`--extra` needs the `=` form**: `--extra="--plant-context"`. Argparse reads
  a dash-leading value as another option.
- **`scp -r seedN dest/`** flattens into `dest/` when `dest` does not exist.
  Files landed one level too high once.
- **A 175-wide observation is ambiguous** between preview and plant context
  (five values each), so `train.py` writes `arms.json` beside each model and
  everything downstream reads it. `rq1_blind` predates this and correctly
  defaults to blind.
- **MuJoCo is not bit-reproducible across CPUs.** Laptop and cluster agreed on
  every conclusion but differed on 33 of 1400 episodes. All reported numbers
  must come from one machine.

## Methodological standards established

Apply these to any new result:

1. **Five seeds minimum.** One seed said p = 0.0026; five said p = 0.125.
2. **Two independent `--scenario-seed` draws.** Three separate results looked
   significant on one draw and vanished on another.
3. **Measure the ceiling first.** The per-scenario hindsight oracle bounds what
   any scheduler can win. `summarise_batch.py` prints it.
4. **Compare tracking only on scenarios both controllers finish.**
5. The fixed baseline scoring identically across seeds is **correct** — it does
   not depend on the training seed.

## Tooling built this session

All committed, all covered by the 118-test suite.

| script | purpose |
|---|---|
| `run_seeds.py` | parallel multi-seed launcher, batch manifest, core-affinity aware |
| `slurm_batch.sh`, `slurm_confirm.sh` | SLURM job scripts |
| `confirm_advantage.py` | paired PPO-vs-fixed comparison, exact McNemar |
| `summarise_batch.py` | multi-seed aggregation, sign test, oracle ceiling, scheduling correlations |
| `analyse_within_episode.py` | within-episode gain behaviour, lead/lag, decisive-episode plots |
| `plot_training.py` | training/evaluation curves with a convergence verdict |
| `dashboard.py` | live 3-panel comparison, hand-tunable gains, plant sliders |
| `replay.py` | MuJoCo viewer replay with wind arrow |

## Open next steps

**Blocked on a supervisor conversation.** `FINDINGS.md` ends with three
options; the user was going to discuss them. Do not start compute on B or C
without confirmation.

- **A.** Write up the negative result as it stands.
- **B.** Raise the 2% ceiling so there is more to win — widen disturbances, or
  give the agent a speed action (needs the baseline to get the same freedom).
  Most likely to yield a positive result.
- **C.** End-to-end RL controller comparison (supervisor's original request).
  Needs a stated compute budget; an undertrained baseline proves nothing.

Smaller items:

- Pull all five seeds' `evaluation_history.csv` / `episodes.csv` /
  `monitor.csv` per arm so `plot_training.py` shows a real seed band (currently
  n=1 per arm locally; files are ~7 KB each).
- `adaptation_scenarios.PhysicsConfig` still has no delay/noise fields, so the
  paired adaptation evaluator runs on the clean plant. Changing it alters
  scenario fingerprints.
- The preview arm (`--preview`) is implemented and tested but has never been
  trained.
- One commit on `main` was pushed with a failing test (panel-count assertion),
  fixed in the next commit.

## Suggested skills

- **`diagnose`** — for any "the result looks wrong" investigation. This project
  has repeatedly had measurement faults masquerading as model failures; the
  reproduce → minimise → hypothesise → instrument loop fits it well.
- **`tdd`** — new analysis scripts here earn their trust from tests. Every
  false positive this session was caught by a test or a check, not by
  inspection.
- **`artifact-design` / `dataviz`** — if producing thesis figures from the CSVs.
- **`stop-slop` / `humanizer`** — when drafting thesis prose.

Do **not** reach for the video, Clerk, or web-app skills; nothing here needs
them.

## Working style that fits this user

- They ask for plain-language explanations and want to genuinely understand the
  control theory and RL, not just receive results. Explain mechanisms.
- They prefer **step-by-step instructions, one step at a time**, when operating
  the cluster — they said so explicitly after a multi-command message.
- They validate cheaply before committing long compute (see
  `feedback-staged-compute` memory).
- They catch real problems: the identical-157 question led directly to the
  second-draw standard that overturned three results. Take their observations
  seriously.
