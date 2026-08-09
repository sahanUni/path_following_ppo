"""Frozen experiment constants for the development PPO path-following study."""

from __future__ import annotations

import numpy as np


SEED = 7

# The path split agreed before implementation. Held-out and stress paths never
# appear in PPO training resets.
# Every path in the catalogue EXCEPT `needle`. Selected on measured sensitivity
# to the gains, not on difficulty -- the two come apart at the extreme. Kp 26 vs
# Kp 50 at 0.7 m/s separates by 26% on hairpin, 26% on zigzag60 and 22% on
# square, all of which were previously unused, while `needle` separates by 3.0%
# because its corners are below the car's achievable turn radius: the demand is
# outside the VEHICLE's envelope, so no gain choice changes the outcome much and
# the agent is punished whatever it does. It stays out for the same reason
# gusts stop at 12 N.
TRAIN_PATHS = (
    "arc",
    "scurve",
    "uturn",
    "slalom",
    "figure8",
    "hairpin",
    "spiral",
    "zigzag",
    "square",
    "zigzag60",
    "zigzag30",
)
HELD_OUT_PATHS = ("figure8", "hairpin", "square", "zigzag60")
STRESS_PATHS = ("zigzag30", "needle")
SPEED_TARGETS = (0.3, 0.5, 0.7)

# Valid plant range established by Path_Following_Sandbox/EXPERIMENT_DESIGN.md.
NOMINAL_MASS_KG = 10.0
NOMINAL_FRICTION = 1.0
NOMINAL_ACTUATOR = 1.0
MASS_RANGE_KG = (5.0, 50.0)
FRICTION_RANGE = (0.2, 1.0)
ACTUATOR_RANGE = (0.4, 1.5)

# Seeded wind gusts (2D Ornstein-Uhlenbeck, world frame). The upper bound is
# chosen to STRADDLE the point where the right answer inverts: measured on the
# arc at 0.3 m/s, raising Kp reduces tracking error up to about 12 N and starts
# making it worse by 15 N. Training only below that boundary would teach the
# single rule "more gain is better", which is exactly the mistake the current
# agent makes. The agent has to see both sides to learn the tradeoff.
GUST_SIGMA_RANGE_N = (2.0, 12.0)
GUST_TAU_RANGE_S = (0.3, 3.0)

# Checkpoint selection runs this fixed gust alongside the calm episodes. A model
# selected purely on still-air tracking cannot be selected FOR adaptation: the
# criterion never sees the conditions adaptation exists to handle. Held constant
# (fixed sigma, tau and seed) so scores stay comparable across checkpoints and
# across runs.
# 6 N, not 10: measured on the hardest episode in the suite (zigzag at 0.7 m/s),
# 10 N is past the cliff -- the car leaves the corridor at every gain from Kp 26
# to 50, so the episode contributes no signal while supplying most of the score.
# At 6 N the same episode finishes and separates Kp 42 from Kp 50 by ~10%, which
# is the sensitivity checkpoint selection needs in order to mean anything.
EVAL_GUST_SIGMA_N = 6.0
EVAL_GUST_TAU_S = 1.0

# Actuator dead time and steering-sensor noise.
#
# WHY THESE EXIST. Without them the plant has no cost for aggression: the car
# reads its cross-track error exactly, the commanded wheel force lands the same
# instant it is computed, and nothing rings. Raising Kp is then monotonically
# better until the wheel clamp truncates the command, so the best fixed gain is
# whatever saturates first, PPO pins its Kp action at the box edge, and the
# Fixed PID baseline matches it exactly. There is no tradeoff to schedule, and a
# gain-scheduling study with no tradeoff has no result to report.
#
# Dead time supplies the stability side of the tradeoff: a correction that lands
# `delay` seconds late arrives after the error it was computed for has changed,
# so it overshoots, and the overshoot grows with Kp. Noise supplies the effort
# side, and is also the only reason Kd is a real decision -- the derivative of a
# clean signal is free information, the derivative of a noisy one is mostly
# amplified garbage.
#
# Both are drawn per episode and appear NOWHERE in the observation, so they are
# hidden plant parameters in exactly the sense mass and friction already are: a
# blind agent must infer them from the error history, a privileged one is told.
#
# Units: delay is stored in seconds and rounded to whole DT (0.002 s) physics
# steps at reset. 0.08 s is 40 physics steps, or four whole PPO control steps.
#
# The cap is measured, not guessed. Swept on `arc` at 0.5 m/s with the v5
# baseline gains, mean error in cm:
#
#   delay   Kp 20   Kp 34   Kp 50
#    0 ms    0.26    0.26    0.25   <- clean plant: bigger Kp always wins
#   50 ms    0.26    0.26    0.25
#  100 ms    0.98    2.65    6.78   <- preference INVERTS, and Kd=0 fails outright
#  200 ms    7.59    fail    fail
#
# Below ~60 ms nothing happens; by 100 ms the whole ranking has flipped; by
# 200 ms most of the action box is unreachable. 0.08 s sits inside the band
# where the tradeoff exists and every gain in the box is still survivable.
ACTUATOR_DELAY_RANGE_S = (0.0, 0.08)
# Metres of white Gaussian noise added to the cross-track error the steering PID
# measures.
#
# Also measured. Noise acts almost entirely through Kd, which is the point --
# the derivative of a clean signal is free information, so on the old plant the
# Kd action channel was very nearly decorative. Same sweep, no delay:
#
#   sigma      Kd 0    Kd 4.2    Kd 8
#   0.0 mm     0.26     0.26     0.26
#   0.5 mm     0.26     0.61     fail    <- Kd now has a real cost
#   1.0 mm     0.27     fail     fail
#   5.0 mm     0.29     fail     fail    <- with Kd=0 it barely registers
#
# Delay and noise therefore pull Kd in OPPOSITE directions: dead time makes
# damping mandatory (Kd=0 fails at 100 ms), noise makes it expensive. That is
# the two-sided tradeoff the gain scheduler exists to resolve. The unfiltered
# derivative in core/pid.py is what makes this so sharp; that PID is a verbatim
# copy of the thesis controller and is deliberately not "improved".
SENSOR_NOISE_RANGE_M = (0.0, 0.0004)

# Held constant for checkpoint selection, same argument as EVAL_GUST_SIGMA_N: a
# model scored on a delay-free, noise-free plant is being selected on the plant
# where high gain is free, which is the opposite of what the study is for.
#
# Chosen by measuring three candidate operating points over the nine reachable
# (Kp, Kd) corner combinations of the v5 box on four conditions, under the eval
# gust. 50 ms / 0.25 mm was too soft -- Kp 50 still won everywhere. 70 ms /
# 0.35 mm started losing episodes, which wastes score on insensitive failures.
# At 60 ms / 0.30 mm all 36 episodes finish, the best-to-worst gain spread
# reaches 5.3x, and the best gain pair DIFFERS across three of the four
# conditions -- so the criterion can tell an adapting policy from one that
# found a better constant, which is the whole reason it exists.
EVAL_ACTUATOR_DELAY_S = 0.06
EVAL_SENSOR_NOISE_M = 0.0003

# A third evaluation tier where FINISHING is in doubt.
#
# The calm and gust tiers both run a 10 kg car on clean floor, and every episode
# in them completes at every reachable gain. That makes the whole selection
# criterion a tracking-only measurement -- and tracking is not the axis the gain
# mostly controls here. Measured over 60 scenarios drawn from the stage-3
# training distribution, fixed Kp 15 completes 48/60 and Kp 50 completes 40/60
# while their mean errors differ by well under a millimetre. Selection that
# cannot see completion cannot see the difference that matters: run v7_seed7
# sat flat for all 40 checkpoints and froze `best_model` at 50k steps.
#
# Tuned to be gain-SENSITIVE rather than merely hard, which is the trap the gust
# cap and EVAL_GUST_SIGMA_N already had to be walked back from. Measured over
# the three eval paths at three speeds, completions out of 9:
#
#   Kp 15 -> 9    Kp 26 -> 9    Kp 34 -> 7    Kp 42 -> 7    Kp 50 -> 6
#
# Graded, and nothing is unwinnable: a good gain choice finishes everything, a
# bad one does not. Tiers that merely raise mass or dead time were rejected --
# they failed 2 of 9 at EVERY gain, which is score with no information in it.
EVAL_STRESS_MASS_KG = 20.0
EVAL_STRESS_FRICTION = 0.35
EVAL_STRESS_ACTUATOR_DELAY_S = 0.07
EVAL_STRESS_SENSOR_NOISE_M = 0.0003

# Paired adaptation evaluation. These values are serialized into each run
# manifest so later changes here cannot silently alter an existing result.
ADAPTATION_MASS_LEVELS_KG = (5.0, 10.0, 50.0)
ADAPTATION_FRICTION_LEVELS = (0.2, 0.6, 1.0)
ADAPTATION_ACTUATOR_LEVELS = (0.4, 1.0, 1.5)
ADAPTATION_MASS_STEPS_KG = ((10.0, 30.0), (30.0, 10.0))
ADAPTATION_FORCE_PULSES_N = (-5.0, -2.5, 2.5, 5.0)
ADAPTATION_EVENT_START_FRACTION = 0.40
ADAPTATION_FORCE_DURATION_S = 0.75
ADAPTATION_PRE_WINDOW_S = 1.0
ADAPTATION_RESPONSE_WINDOW_S = 2.0
ADAPTATION_RECOVERY_RELATIVE_TOLERANCE = 0.25
ADAPTATION_RECOVERY_FLOOR_M = 0.02
ADAPTATION_RECOVERY_SUSTAINED_S = 0.5
ADAPTATION_MEANINGFUL_IMPROVEMENT_PCT = 5.0

SPEED_PID_GAINS = np.array([3.0, 1.0, 0.0], dtype=np.float64)

# One PPO decision covers ten 0.002 s MuJoCo/PID steps.
FRAME_SKIP = 10
HISTORY_LENGTH = 10

# Reward terms are kept individually in info/trace. Their initial magnitudes
# are deliberately conservative and can be audited before a long run.
PROGRESS_WEIGHT = 5.0
# Weight of the SATURATING tracking cost below, which is bounded in [0, 1] per
# step. Small because it is paid on every one of up to ~2000 control steps.
TRACKING_WEIGHT = 0.035
# The length scale tracking error is measured against, set to the size of a GOOD
# run's error (~0.4 cm) so the cost is O(1) exactly where controllers need to be
# told apart.
#
# The cost is deliberately saturating -- z/(1+z) with z = (d/scale)^2 -- and not
# a plain square. Two failures bracket the right answer here:
#
#   * normalising by the corridor half-width (1.0 m) made the cost ~1.6e-5 per
#     step, so a 26% tracking improvement was worth 0.03% of episode reward and
#     PPO could not see it at all;
#   * a plain square at a 0.05 m scale fixed the sensitivity and broke the tail
#     instead. The corridor only terminates at 1.0 m, so a car limping 55 cm off
#     path under heavy disturbance survives the full ~2000-step time limit at
#     -200/step. Worst observed episode: -475,734 against a median of +42. No
#     value function fits that, and training degraded.
#
# Saturating keeps sub-centimetre discrimination while bounding a whole bad
# episode near the failure penalty, so excursions cost a lot without swamping
# every other term.
TRACKING_SCALE_M = 0.005
ACTION_SMOOTHNESS_WEIGHT = 0.01
FINISH_BONUS = 20.0
FAILURE_PENALTY = 20.0

# A full path must finish within twice its ideal traversal time plus 5 seconds.
TIME_LIMIT_FACTOR = 2.0
TIME_LIMIT_MARGIN_S = 5.0

# Curriculum fractions refer to the requested total PPO timesteps.
CURRICULUM_NOMINAL_END = 0.20
# Stage 3 is the only stage that serves disturbances. At the previous 0.70 the
# agent met wind for the last 30% of training, and gusts (one kind of three) for
# roughly 10% of it -- far too little to learn a condition-dependent gain rule.
CURRICULUM_RANDOMIZED_END = 0.40

# Fixed normalizers keep evaluation independent from training statistics.
E_CT_SCALE_M = 1.0
E_CT_RATE_SCALE_MPS = 2.0
INTEGRAL_SCALE_MS = 1.0
SPEED_SCALE_MPS = 1.5
YAW_RATE_SCALE_RADPS = 10.0
MAX_TARGET_SPEED_MPS = max(SPEED_TARGETS)

# The calibration search space is not the PPO action box. The latter is
# derived from the stable/useful candidates and saved to calibration.json.
CALIBRATION_SEARCH_LOW = np.array([0.25, 0.0, 0.0], dtype=np.float64)
CALIBRATION_SEARCH_HIGH = np.array([50.0, 5.0, 10.0], dtype=np.float64)
MIN_GAIN_SPAN = np.array([2.0, 0.5, 0.5], dtype=np.float64)
# MIN_GAIN_SPAN measures total width, which is not enough on its own: if the
# winning candidate is also the most extreme survivor on some axis, that axis
# collapses onto the baseline (high == base) while the total width still looks
# healthy. The action channel then maps to nothing and the policy cannot move
# that gain at all. This is the per-side floor, as a fraction of the baseline,
# so every action channel keeps authority in BOTH directions.
MIN_GAIN_MARGIN = 0.25

# Calibration ranks completion before tracking, which is right until completion
# stops meaning what it looks like. On the dead-time plant a very low gain
# finishes every scenario by crawling: measured over the v6 sweep, Kp 1.23
# completed 99/99 at 5.25 cm mean error while Kp 20.32 completed 93/99 at
# 1.88 cm and beat it head to head on 93 of 93 shared scenarios, 5.46x. The six
# extra completions came from ONE adverse physics draw (44 kg, friction 0.20)
# where the crawler "succeeded" at 9-29 cm inside a 1.0 m corridor.
#
# So completion is treated as a BAND rather than a strict ordering: anything
# within this fraction of the best completion count is a tie, and tracking
# decides. The value is not arbitrary -- one physics draw is 6/99 = 6.1% of the
# scenario set, so the slack has to clear roughly 7% to stop a single sample
# from dictating the result. The selection is stable anywhere from 0.90 to 0.80.
COMPLETION_TOLERANCE = 0.90

# Initial PPO settings mirror standard SB3 PPO defaults while making the MLP
# explicit for experiment reproducibility.
PPO_KWARGS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": [64, 64]},
}
