# Findings: online RL tuning of a steering PID

Status as of 2026-08-11. Written for discussion, not as a chapter draft.

Everything below is reproducible from this repository. Commands are in
[README.md](README.md) and [SERVER.md](SERVER.md); the numbers come from
`runs/rq1_blind`, `runs/rq2_context` and `runs/v7_seed7`.

---

## The question

A car follows a path using a PID controller on its steering. A PID has three
numbers — Kp, Ki, Kd — that are normally chosen once, by hand, and then left
alone. The question is whether a reinforcement learning agent can do better by
adjusting them online, in response to conditions the fixed controller cannot
see: how heavy the car is, how slippery the floor is, how much lag the
actuators have.

## Summary

1. The original simulation could not answer the question, because in it a
   higher gain was always better. We diagnosed this and fixed it.
2. On the corrected simulation, the choice of gain mostly decides **whether the
   car finishes the course**, not how neatly it drives.
3. We measured the ceiling: choosing the perfect gain for every scenario, with
   hindsight, is worth **2% of episodes**. That is the most any gain scheduler
   can win here.
4. The trained agent **matches** a carefully hand-calibrated fixed PID. It does
   not reliably beat it, and every apparent win disappeared when re-tested on a
   fresh set of evaluation scenarios.
5. Giving the agent the hidden plant parameters directly did **not** help. It
   also does not adapt its gain to anything measurable, even when it wins.

The defensible claim is that **RL reaches the performance of a carefully
calibrated PID without requiring the calibration** — not that it exceeds it.

---

## What we did, and why

### 1. Why the first experiments showed nothing

PPO and the fixed PID produced identical trajectories. The cause was not a bug
in the agent. In the original simulation the car measured its position error
exactly, and its wheel commands took effect instantly.

In a world like that, steering harder is always better, right up to the point
where the wheels saturate. There is no such thing as too aggressive, so the
best setting is simply "as high as possible". Both controllers found it. The
agent had correctly solved a problem with no interesting answer.

**This is worth stating plainly in the thesis**: a gain-scheduling study needs a
plant where the best gain is a compromise. Ours was not.

### 2. Making the simulation realistic

We added two imperfections that every real robot has:

- **Actuator dead time.** The wheel command arrives 0–80 ms late. A correction
  computed for an error that has since changed causes overshoot, and the
  overshoot grows with Kp. This puts a stability ceiling on the gain.
- **Sensor noise.** The measured cross-track error is slightly wrong (0–0.4 mm).
  The D term reacts to how fast the error changes, so with noise it mostly
  reacts to noise. This makes Kd a real decision.

Both were measured, not guessed. On one path, mean error in cm:

| dead time | Kp 20 | Kp 34 | Kp 50 |
|---|---|---|---|
| 0 ms | 0.26 | 0.26 | **0.25** |
| 100 ms | **0.98** | 2.65 | 6.78 |

The ranking inverts. And noise acts almost entirely through Kd: at 0.5 mm,
Kd = 0 gives 0.26 cm, Kd = 4.2 gives 0.61 cm, Kd = 8 fails outright.

So dead time makes damping **necessary** and noise makes it **expensive** — a
genuine two-sided trade-off, which is the precondition the whole question
depends on.

Two rules the implementation keeps, both worth defending:

- The controller sees the noisy error; the **score** uses the true one.
  Otherwise we would be measuring performance that did not happen.
- Neither the dead time nor the noise level appears in the agent's
  observation. They are hidden, exactly like mass and friction.

### 3. Calibrating a fair baseline

The comparison is only meaningful against a well-tuned PID, so this took three
attempts and two of the failures are instructive.

- A 48-candidate random search picked a gain that "succeeded" everywhere by
  barely steering at all — wandering up to 29 cm off a path with a 1 m corridor.
  It completed 99/99 scenarios at 5.25 cm mean error, beating a gain that
  completed 93/99 at 1.88 cm. **Cause:** the ranking put completion strictly
  above accuracy, so six extra completions outranked a 2.8× better controller.
  Fixed by treating completion as a band rather than an absolute ordering.
- The same random search put only **one** of its 48 candidates in the region
  that turned out to matter. We replaced it with a targeted grid.

Final baseline: **Kp 26, Ki 0.5, Kd 4.4**, with the agent allowed to move Kp
between 15 and 50. That range was set from measurements of where the best gain
actually sits under different conditions, and is documented as a deliberate
choice rather than presented as derived.

### 4. What the gain actually controls

An important and initially surprising result: **across conditions, the gain
barely affects how neatly the car drives, but strongly affects whether it
finishes at all.**

Tracking differences between gain settings are fractions of a millimetre.
Completion differences are around 20% of episodes.

This mattered practically. Our evaluation had been scoring only tidiness, which
is why the learning curves were flat for 40 checkpoints — the score could not
see the thing that was improving. We added a harder evaluation tier where
finishing is genuinely in doubt, tuned so that a good gain finishes 9 of 9 and a
bad one finishes 6 of 9.

---

## Results

### The first result looked strong — and was not real

One training run scored **173/200 completions against 158** for the fixed PID,
p = 0.0026.

Five training runs told a different story:

| seed | agent | fixed PID | p |
|---|---|---|---|
| 21 | 173/200 | 157 | 0.0004 |
| 7 | 166/200 | 157 | 0.078 |
| 123 | 159/200 | 157 | 0.73 |
| 42 | 158/200 | 157 | 1.00 |
| 84 | 157/200 | 157 | 1.00 |

One of five is significant. Across seeds, p = 0.125. The seed-to-seed spread is
about three times the average advantage.

We then re-tested the winning run on a **different set of 200 evaluation
scenarios**. Its margin fell from 16 episodes (p = 0.0004) to 5 (p = 0.18).
The apparent win was a lucky training seed and a lucky evaluation set stacked
together.

*(This check was prompted by a good question: why did every seed report exactly
157/200 for the fixed PID? Answer: by design — the baseline does not depend on
the training seed, so all five scored the identical controller on the identical
episodes. But that also meant one evaluation draw was common to all five
comparisons, which is what led us to test a second one.)*

### Giving the agent the answer did not help

We trained five more runs with mass, friction, actuator strength, dead time and
sensor noise **written directly into the observation**.

| | mean completion | spread |
|---|---|---|
| blind | 0.813 | ± 0.034 |
| given the plant parameters | 0.807 | ± 0.086 |

No better on average, and two and a half times more variable — one run
collapsed to 134/200, worse than every fixed gain we tested. So the limitation
is **not** that the blind agent cannot work out the conditions.

### The agent does not adapt, even when it wins

For each episode we compared the gain the agent chose against the dead time it
was actually facing. An adapting controller would use lower gains when the lag
is high — a clear negative relationship.

All ten runs came out at essentially zero (+0.19 to −0.13), including the runs
given the answer directly.

Looking **inside** episodes, on the best policy in the project, the gain
correlates with nothing measurable — error, error rate, heading error, speed,
yaw rate, wheel saturation, curvature under the car, curvature ahead: all within
±0.20. And episodes it wins look the same as episodes everything wins.

Instead, the gain swings across most of its allowed range in every episode and
changes by about 4.5 units every 20 ms. That is undirected dithering, not a
schedule.

### Why: the ceiling is only 2%

This is the number that explains everything above.

| controller | completed |
|---|---|
| best single fixed gain | 160/200 |
| **perfect gain for each scenario, chosen with hindsight** | **164/200** |

Choosing the ideal gain for every scenario is worth **4 episodes in 200**. That
is the entire prize available to anything that picks a gain per episode — and it
is smaller than the variation between training runs.

So: the agent shows no benefit from knowing the conditions because knowing the
conditions is worth almost nothing here. Every null result in this project has
this one cause.

---

## What holds up

Tested on two independent evaluation draws, the agent:

- **significantly beats** poorly chosen gains (Kp 34, 42, 50) every time;
- is **never significantly worse** than the carefully calibrated baseline;
- **does not reliably beat** that baseline.

So the honest statement is:

> Online RL adjustment of PID gains reaches the performance of a carefully
> calibrated fixed PID without requiring the calibration — which here cost a
> failed 48-candidate search, a targeted 10-candidate search, and a hand-set
> range. It does not exceed that performance, and it does not adapt to
> conditions.

---

## Methodological points worth defending

- **Multiple seeds.** One run said p = 0.0026. Five said p = 0.125.
- **Two evaluation draws.** Three separate results looked significant on one
  set of scenarios and vanished on another.
- **Paired comparisons.** Every controller meets identical scenarios, so
  difficulty cancels; significance is tested only on episodes where the two
  controllers disagree.
- **The ceiling measured before the claim.** Establishing that only 2% was
  available prevents over-reading a 2% result.
- **Tracking compared only on episodes both controllers finish.** Comparing each
  controller's average over its own completions flatters whichever one gives up
  on the hard episodes. This correction reversed one of our earlier conclusions.

---

## Options

**A. Write up as it stands.** A thorough negative result with a modest positive,
plus a measured explanation of why. Honest, complete, defensible.

**B. Raise the ceiling, then re-test.** The 2% prize is the obstacle. Two ways
to raise it, both principled rather than result-fishing:

- widen the disturbance range until fixed gains genuinely fail more often;
- let the agent also control target speed. Speed affects completion far more
  than gain does, so "slow down before a tight corner" is a much larger prize
  than "adjust Kp". This needs the fixed baseline to get the same freedom, or
  the comparison is unfair.

**C. Compare against an end-to-end RL controller** that drives the car directly
instead of tuning a PID. Worth doing with a stated compute budget; an
undertrained end-to-end baseline would prove nothing.

The scoping decision is yours. Option B is the one most likely to turn this into
a positive result, and it is a change to the experiment rather than to the
analysis, so it does not compromise anything above.
