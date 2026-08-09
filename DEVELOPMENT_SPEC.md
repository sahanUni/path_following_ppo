# Development specification

This document records the design agreed before implementation. Changes to
these decisions should be explicit experiments rather than silent edits.

## Scientific comparison

The development claim is: **a causal PPO steering-PID gain scheduler using
available controller measurements versus one calibrated fixed-gain PID**.
The fixed PID and PPO use the same plant, speed controller, path reference,
actuator allocator, termination logic, and evaluation metrics.

Only one seed (`7`) is used during development. Multiple training seeds are
required before making statistical or thesis-level performance claims.

## Rates and actions

The plant and PID loops run at 500 Hz. PPO acts at 50 Hz; one action is held for
ten physics/PID steps. Its normalized action lies in `[-1, 1]^3` and maps
piecewise to bounded absolute gain targets:

- `-1 -> calibrated lower bound`
- `0 -> calibrated global fixed-PID baseline`
- `+1 -> calibrated upper bound`

Version one has no hard gain slew limiter. Action-change cost discourages
chattering, and every target/applied gain is logged.

## Observation

One frame contains normalized causal values:

1. cross-track error
2. cross-track-error rate
3. steering integral state
4. heading error
5. forward speed
6. yaw rate
7. requested steering command
8. applied steering command
9. applied speed command
10. previous normalized Kp action
11. previous normalized Ki action
12. previous normalized Kd action
13. commanded speed
14. wheel utilization
15. steering PID saturation flag
16. downstream allocation-limited flag
17. steering integrator-hold flag

Ten frames are concatenated, giving 0.2 seconds of history. Hidden physics,
path identity, path progress, true distance, future waypoints, and curvature
are excluded. A context teacher and preview agent are deferred.

## Speed and actuator allocation

The commanded speed is fixed per episode at 0.3, 0.5, or 0.7 m/s. The fixed
speed PID runs at 500 Hz. The steering-priority allocator enforces
`|v_applied| + |omega_applied| <= 1`, so actual speed naturally falls when a
corner consumes wheel authority. This slowdown is logged as an outcome; PPO
does not control the speed PID.

## Reference, reward, and termination

The closest waypoint is selected only from the previous reference index and
the next 200 path samples; the index never moves backward. Cross-track error
drives the steering PID. True Euclidean distance to that reference point drives
the reward and headline tracking metrics because cross-track projection can
understate overshoot at tight corners.

Per-step reward:

```text
5.0 * progress_delta
- 2.0 * mean((true_distance / 1 m)^2)
- 0.01 * ||action_t - action_(t-1)||^2
+ 20 on finish
- 20 on corridor exit, invalid state, or time-limit failure
```

Episodes start exactly on the first path point and aligned with `+x`; the pose
is never randomized. Success is reaching the final reference point. Corridor
exit or invalid state terminates the episode. A path-length/speed-dependent
budget truncates non-finishing episodes.

## Curriculum and ranges

1. First 20%: nominal plant, `arc/scurve/uturn`, speeds 0.3/0.5.
2. Next 50%: all training paths and speeds, per-episode randomized physics.
3. Final 30%: same randomization plus a mid-episode mass or external-force event.

Physics ranges:

- mass: 5–50 kg
- friction: 0.2–1.0
- actuator scale: 0.4–1.5

## Metrics and selection

Completion is always considered before tracking quality. Recorded diagnostics
include mean/max true distance, ITAE, completion time, progress, saturation,
allocation limiting, action change, reward components, and gain trajectories.
The best PPO checkpoint is selected lexicographically by completion count and
then mean true distance on a fixed development suite—not by training reward.

## Deferred work

- teacher/context PPO
- curvature or waypoint preview
- learned/free speed and Pareto evaluation
- hard gain slew limiting
- multi-seed statistical evaluation
- interactive results dashboard

