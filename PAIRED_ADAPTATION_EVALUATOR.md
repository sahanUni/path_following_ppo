# Paired Adaptation Evaluator and Results Dashboard

## Status

Implemented on 2026-08-04. The typed scenario contract, deterministic paired
runner, shared adaptation metrics, immutable run directories, and read-only
dashboard are implemented in the files described below.

The four-scenario integration run is stored under
`runs/adaptation_evaluation/implementation_smoke_v2`. It completed all four pairs,
verified identical event timing for both controllers, reproduced the previous
nominal evaluator exactly, and confirmed that loading/evaluating PPO did not
change the model artifact.

## Recommendation: extend the current project

Implement this work inside `Path_Following_PPO`, not as a fresh project.

The comparison is only trustworthy if both controllers use the same `PathFollowingEnv`, path definitions, reset pose, timing, actuator allocation, calibrated fixed gains, trained PPO checkpoint, and metric calculations. A second project would duplicate these components and create a risk that small simulator or metric differences are mistaken for controller improvements.

The new functionality should still be isolated:

```text
Path_Following_PPO/
  adaptation_scenarios.py
  adaptation_evaluate.py
  adaptation_metrics.py
  dashboard.py                 # or dashboard/ if it grows
  runs/
    adaptation_evaluation/
      <run_id>/
```

The evaluator is a new experiment within the existing project, not a modification of training.

## 1. Question being tested

Does the frozen causal PPO gain scheduler adapt better than the calibrated fixed-gain PID when the vehicle or disturbance changes unexpectedly?

The comparison is:

- **Fixed controller:** the calibrated fixed steering PID.
- **Adaptive controller:** the frozen PPO policy that schedules steering `Kp`, `Ki`, and `Kd` from current and historical controller measurements.
- **Pairing:** both controllers receive the exact same path, target speed, initial pose, seed, physical parameters, and timed disturbance.
- **Blindness:** the PPO policy is not told the path identity, future waypoints, hidden mass/friction values, or future disturbances.
- **Speed:** target speed remains fixed per episode. The existing speed controller and steering-priority allocator determine achieved speed.

This evaluator must not retrain or fine-tune the PPO model.

## 2. Why this evaluator is needed

The completed nominal comparison shows that both methods follow the paths successfully, but their differences are extremely small:

- Training paths: PPO mean distance was about `0.096%` lower and ITAE about `0.281%` lower, while maximum distance was about `0.963%` higher.
- Held-out paths: PPO mean distance was about `0.033%` higher, ITAE about `0.154%` lower, and maximum distance about `3.225%` higher.
- Completion and achieved speed were effectively identical.
- The PPO policy kept `Kp` almost permanently at its upper bound and changed `Ki` and `Kd` rapidly.

These nominal results do not demonstrate useful online adaptation. The paired evaluator focuses on the situation where scheduling gains could have a real advantage: a change in plant dynamics or an external disturbance.

Existing results remain under `runs/`; the experiment history and original controller contract are documented in `DEVELOPMENT_SPEC.md` and `README.md`.

## 3. Fair-comparison rules

Every result must satisfy all of the following:

1. A scenario has a stable, unique `scenario_id`.
2. Fixed PID and PPO run from the same serialized scenario definition.
3. Both use the same deterministic reset seed and unchanged starting pose.
4. Both receive the same disturbance at the same simulation time—not when they reach a controller-dependent waypoint.
5. The PPO checkpoint and calibration file are loaded once and recorded by path and file hash.
6. The policy is deterministic during evaluation.
7. The evaluator rejects incomplete pairs instead of silently averaging them.
8. Metrics are calculated by shared code from saved traces, not independently inside each controller.
9. Reward is not used as the headline comparison metric.
10. No controller may receive information unavailable to it in the agreed controller interface.

The first development version uses one seed, matching the current development decision. Multi-seed validation is a later phase after the evaluator works correctly.

## 4. Scenario definition

Scenarios should be explicit data records rather than combinations hidden in loops. A manifest makes each comparison reproducible.

Suggested schema:

```yaml
scenario_id: slalom_v05_mass_10_to_30
path_key: slalom
target_speed: 0.5
seed: 0
base_physics:
  mass: 10.0
  friction_scale: 1.0
  actuator_scale: 1.0
event:
  kind: mass_step          # none | mass_step | force_pulse
  start_time_s: 4.0
  end_time_s: null
  value: 30.0
tags: [development, dynamic, mass]
```

The manifest for a run must contain the fully expanded records actually executed. It must not depend on later changes to Python defaults.

### 4.1 Initial development matrix

Keep the first matrix intentionally small enough to inspect trace by trace.

Representative paths:

- Development: `arc`, `slalom`, `zigzag`, `spiral`
- Held-out: `figure8`, `hairpin`, `zigzag60`

Target speeds:

- `0.3`, `0.5`, and `0.7`

Scenario families:

- Nominal control case with no event.
- Stationary mass variations spanning low, nominal, and high values within the environment's supported bounds.
- Stationary friction variations spanning low, nominal, and high scales.
- Stationary actuator-strength variations spanning weak, nominal, and strong scales.
- Mid-episode mass steps in both directions.
- Mid-episode lateral force pulses in both directions and at two magnitudes within the existing force limits.

Use one-axis-at-a-time variations first. A small set of deliberately selected corner cases can be added after those results are understood; do not begin with a full factorial grid.

The exact numeric stationary grid and total episode count are implementation-time decisions and must be written into the generated manifest. Existing environment bounds must be checked before finalizing them.

### 4.2 Event timing

Dynamic events are time-based so the two controllers experience the change at the same instant. For paths with different expected durations, event timing may be derived once from nominal path length and target speed, then stored as absolute seconds in the scenario manifest.

Each dynamic episode must contain enough time for three analysis windows:

1. **Pre-event:** establish normal tracking error.
2. **Response:** capture the immediate error and control response.
3. **Recovery:** determine whether and when tracking returns near its pre-event level.

The saved trace must include an event-active flag and the applied mass/force value, so event timing can be verified rather than inferred.

### 4.3 Deferred disturbance types

The current environment already supports mass changes and force events. Mid-episode friction and actuator-strength changes should be deferred unless implementing them is simple and testable. Stationary friction and actuator variations are sufficient for the first version.

## 5. Metrics

### 5.1 Existing whole-episode metrics

- Completion and failure reason
- Mean path distance
- Maximum path distance
- ITAE
- Episode duration
- Mean achieved speed
- Steering saturation fraction
- Speed-allocation/steering-priority fraction
- Gain statistics and gain-change statistics for PPO

ITAE means **Integral of Time-weighted Absolute Error**. In simple terms, tracking errors become more costly the longer they persist, so a controller that recovers quickly is rewarded relative to one that keeps the same error for a long time.

### 5.2 Adaptation-specific metrics

For dynamic scenarios, calculate:

- Median and peak tracking error in the pre-event window
- Peak tracking error after the event
- Post-event integrated absolute error
- Post/pre error ratio
- Recovery time
- Failure to recover before episode end
- PPO gain change after the event
- Time to first meaningful PPO gain response
- Gain settling or continued gain chatter

The exact recovery threshold must be declared before the main evaluation. A sensible initial definition is the first sustained interval in which tracking error returns within a tolerance band based on the pre-event error plus an absolute floor. The sustained interval and floor must be constants in configuration and recorded in the manifest.

### 5.3 Paired reporting

For every scenario, report `PPO - fixed PID` deltas and make the sign convention visible. Aggregate only complete pairs.

Priority order:

1. Completion/safety
2. Maximum error and recovery failure
3. Recovery time
4. ITAE and mean error
5. Saturation, allocation, and gain smoothness

Tiny numerical differences must not be described as meaningful improvements. Before the final experiment, define a practical improvement threshold—for example, a minimum percentage improvement in recovery time or ITAE while completion and maximum error do not worsen. This is a decision threshold, not a statistical-confidence claim; one seed is insufficient for statistical generalization.

## 6. Output contract

Each evaluation creates an immutable run directory:

```text
runs/adaptation_evaluation/<run_id>/
  manifest.json
  episode_summary.csv
  paired_summary.csv
  aggregate.json
  traces/
    <scenario_id>__fixed.csv
    <scenario_id>__ppo.csv
  plots/
  logs/
```

`manifest.json` should record:

- Expanded scenario definitions
- Evaluation configuration
- PPO model path and SHA-256 hash
- PID calibration path and SHA-256 hash
- Relevant source revision when available
- Python/package versions
- Start/end timestamps
- Completion status and missing pairs

`aggregate.json` is the stable dashboard input. The dashboard should not need to understand internal Python objects or recompute simulation results.

Trace rows should include at least time, position, heading, path error, heading error, speed, target speed, wheel commands, saturation/allocation flags, event state, current physical/event value, controller type, and current gains.

## 7. Dashboard

Build a separate, read-only Dash application in the same project. Its first responsibility is inspecting completed evaluation artifacts; it should not train a model or silently run simulations during page load.

### 7.1 Layout

Use the following views:

1. **Overview**
   - Run selector and run-status banner
   - Model/calibration identity badges
   - Completion, mean-error delta, maximum-error delta, ITAE delta, and recovery-time KPI cards
   - Clear warning when the run has missing pairs or only one seed

2. **Scenario Explorer**
   - Filters for path, speed, disturbance family, severity, and outcome
   - Paired scenario table with fixed, PPO, absolute delta, and percentage delta
   - Trajectory overlay showing reference path, fixed PID, and PPO
   - Tracking error and speed against time, with disturbance interval shaded
   - Wheel commands and saturation indicators

3. **Adaptation**
   - Recovery-time and peak-error comparisons
   - Paired-delta heatmap by path and scenario
   - Pre-event, response, and recovery windows
   - Filters that separate stationary variations from mid-episode events

4. **Gain Behavior**
   - PPO `Kp`, `Ki`, and `Kd` against time
   - Gain-bound occupancy
   - Gain-change/chatter summaries
   - Event marker aligned across error and gain plots

5. **Training History**
   - Existing 25k/50k/75k/100k evaluation history as context only
   - Clear separation between training evaluations and adaptation results

6. **Run Metadata**
   - Full manifest, package versions, paths, hashes, and warnings

### 7.2 Dashboard behavior

- Read generated JSON/CSV artifacts; do not rerun evaluation in ordinary callbacks.
- Keep fixed and PPO colors consistent in every chart.
- Preserve units and sign conventions in labels and tooltips.
- Allow direct selection of the worst PPO regression and best PPO improvement.
- Show missing or malformed data explicitly instead of replacing it with zero.
- Remain usable while a run is incomplete by showing only validated pairs and a prominent incomplete-run warning.
- Provide a simple command such as `python dashboard.py --runs-dir runs/adaptation_evaluation`.

A future explicit **Run comparison** control can launch evaluation, but it is outside the first dashboard version because execution, progress reporting, cancellation, and partial-result handling add a separate set of concerns.

## 8. Implementation phases

### Phase 1: scenario and artifact contracts

- Add typed scenario definitions and validation.
- Generate an explicit manifest for the development matrix.
- Add run-directory and file-hash helpers.
- Unit-test serialization, IDs, event timing, and validation.

### Phase 2: paired evaluator

- Extend environment reset/options only where required by supported scenarios.
- Execute fixed PID and frozen PPO from each identical scenario.
- Save complete traces and episode summaries.
- Fail or flag the run when a pair is missing.
- Verify that nominal scenarios reproduce the existing evaluator within numerical tolerance.

### Phase 3: adaptation metrics and aggregation

- Calculate windowed response/recovery metrics from saved traces.
- Produce paired deltas and aggregates.
- Add tests using small synthetic traces with known recovery behavior.
- Generate `aggregate.json` for the dashboard.

### Phase 4: read-only dashboard

- Implement the six views above.
- Add empty, incomplete, malformed, and complete-run states.
- Test run discovery and artifact parsing independently of Dash callbacks.

### Phase 5: development evaluation

- Run the one-seed development matrix.
- Inspect worst/best pairs and verify disturbance traces manually.
- Decide whether the PPO demonstrates practically useful adaptation.
- Only then consider more seeds, new paths, action-bound changes, smoothing, or further training.

## 9. Acceptance criteria

The first implementation is complete when:

- Every executed scenario has exactly one validated fixed/PPO pair.
- The manifest fully reproduces scenario parameters and event timing.
- Starting position and pose remain unchanged.
- Nominal results agree with the current evaluator within an explained tolerance.
- Dynamic traces prove that both controllers received the same event at the same time.
- No PPO weights are changed by evaluation.
- Adaptation metrics pass synthetic unit tests.
- Summary files can be inspected without launching the dashboard.
- The dashboard loads completed and partial runs without crashing.
- The dashboard makes one-seed and incomplete-pair limitations visible.

## 10. Decisions for the next chat

Before or during Phase 1, finalize:

1. The exact low/nominal/high values for stationary mass, friction, and actuator strength.
2. The exact mass-step and force-pulse values and timing.
3. The recovery tolerance, sustained recovery interval, and practical-improvement threshold.
4. Whether the first matrix should use all seven representative paths or begin with a smaller smoke-test subset.

These decisions do not require a new repository. They should become configuration values and be recorded in every run manifest.
