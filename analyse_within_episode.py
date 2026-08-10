"""What does the policy actually DO with the gain inside an episode?

The per-scenario oracle -- the best FIXED gain for each scenario, chosen with
hindsight -- completes 164/200 on this distribution, and the best policy
completes 173. Those extra episodes cannot come from picking a better gain per
episode, because the oracle already picks the best one. They can only come from
changing the gain DURING the episode.

That is the one mechanism the data supports, and nothing so far has described
it. This script does:

  1. find the DECISIVE scenarios -- the policy finishes, no fixed gain does;
  2. replay each with the policy and with the calibrated baseline, keeping the
     full 50 Hz trace;
  3. correlate the policy's instantaneous Kp against instantaneous state, so
     the rule it learned can be stated rather than guessed at;
  4. check whether Kp LEADS or FOLLOWS the error, which separates anticipation
     from reaction;
  5. plot each decisive episode, marking where the baseline left the corridor.

Correlations are computed per episode and then averaged, never by pooling every
timestep. Pooling would let one long episode dominate, and would count
1000 highly autocorrelated samples as 1000 independent ones.

    python analyse_within_episode.py \
        --model runs/v7_seed7/final_model.zip \
        --calibration runs/calibration_v7.json \
        --from-confirmation runs/v7_seed7/confirm_200.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from calibration import GainCalibration
import config
from confirm_advantage import resolve_arms, sample_scenarios
from core import paths
from env import PathFollowingEnv
from rollout import fixed_policy, run_episode


# Instantaneous quantities the gain might plausibly respond to. All are
# available to the policy through its observation history, so a correlation
# here is a rule it could actually have learned -- not a coincidence with
# something it cannot see.
FEATURES = {
    "|e_ct|": lambda rows, extra: np.abs(_col(rows, "e_ct")),
    "e_ct rate": lambda rows, extra: np.abs(np.gradient(_col(rows, "e_ct"))),
    "|e_theta|": lambda rows, extra: np.abs(_col(rows, "e_theta")),
    "speed": lambda rows, extra: _col(rows, "speed"),
    "|yaw rate|": lambda rows, extra: np.abs(_col(rows, "yaw_rate")),
    "wheel util": lambda rows, extra: _col(rows, "wheel_utilization"),
    "saturated": lambda rows, extra: _col(rows, "pid_saturated"),
    "alloc limited": lambda rows, extra: _col(rows, "allocation_limited"),
    "|curvature now|": lambda rows, extra: np.abs(extra["curvature_now"]),
    "|curvature ahead|": lambda rows, extra: np.abs(extra["curvature_ahead"]),
}


def _col(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows], dtype=np.float64)


def curvature_features(rows: list[dict[str, Any]], path_key: str) -> dict[str, np.ndarray]:
    """Curvature under the car, and one metre ahead of it."""
    path = paths.get(path_key)
    kappa = paths.curvature_of(path)
    progress = _col(rows, "progress_s")
    now = np.array([paths.preview_at(path, kappa, s, (0.0,))[0] for s in progress])
    ahead = np.array([paths.preview_at(path, kappa, s, (1.0,))[0] for s in progress])
    return {"curvature_now": now, "curvature_ahead": ahead}


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 10 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def lead_lag(gain: np.ndarray, error: np.ndarray, max_lag: int = 25) -> tuple[int, float]:
    """Lag at which Kp best matches |error|, in control steps (0.02 s each).

    A POSITIVE lag means the gain change comes AFTER the error moved -- the
    policy is reacting. A negative lag means the gain moved first, which would
    be anticipation, only possible from something that leads the error such as
    curvature ahead.
    """
    if len(gain) < 3 * max_lag or gain.std() < 1e-9 or error.std() < 1e-9:
        return 0, float("nan")
    best_lag, best_value = 0, 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = gain[lag:], error[: len(error) - lag] if lag else error
        else:
            a, b = gain[: len(gain) + lag], error[-lag:]
        value = safe_corr(a, b)
        if np.isfinite(value) and abs(value) > abs(best_value):
            best_lag, best_value = lag, value
    return best_lag, best_value


def load_outcomes(path: Path) -> dict[str, dict[int, int]]:
    outcomes: dict[str, dict[int, int]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            outcomes.setdefault(row["controller"], {})[int(row["scenario"])] = int(
                row["finished"]
            )
    return outcomes


def replay(
    calibration: GainCalibration,
    options: dict[str, Any],
    arms: dict[str, bool],
    model: PPO | None = None,
    gains: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = PathFollowingEnv(
        calibration=calibration,
        training=False,
        path_keys=(options["path_key"],),
        fixed_gains=gains,
        preview=arms["preview"],
        plant_context=arms["plant_context"],
    )

    def ppo_policy(observation: np.ndarray) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    try:
        return run_episode(
            env,
            fixed_policy if gains is not None else ppo_policy,
            seed=config.SEED,
            options=options,
        )
    finally:
        env.close()


def plot_episode(
    index: int, scenario: dict[str, Any],
    ppo_rows: list[dict[str, Any]], fixed_rows: list[dict[str, Any]],
    calibration: GainCalibration, output: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t_ppo, t_fixed = _col(ppo_rows, "t"), _col(fixed_rows, "t")

    axes[0].plot(t_ppo, _col(ppo_rows, "dist") * 100, label="PPO", color="#c792ea")
    axes[0].plot(t_fixed, _col(fixed_rows, "dist") * 100, label="Fixed PID",
                 color="#4fc3f7", linestyle="--")
    axes[0].axhline(100.0, color="#ff7a7a", linewidth=1, linestyle=":")
    axes[0].axvline(t_fixed[-1], color="#ff7a7a", linewidth=1)
    axes[0].annotate("baseline leaves corridor", (t_fixed[-1], 60),
                     color="#ff7a7a", fontsize=8, ha="right")
    axes[0].set_ylabel("distance off path (cm)")
    axes[0].legend(fontsize=8)

    axes[1].plot(t_ppo, _col(ppo_rows, "kp"), color="#c792ea", label="PPO Kp")
    axes[1].axhline(float(calibration.base[0]), color="#4fc3f7", linestyle="--",
                    label="fixed Kp")
    for bound in (calibration.low[0], calibration.high[0]):
        axes[1].axhline(float(bound), color="#2c3a57", linewidth=1, linestyle=":")
    axes[1].set_ylabel("Kp")
    axes[1].legend(fontsize=8)

    axes[2].plot(t_ppo, np.abs(_col(ppo_rows, "e_ct")) * 100, color="#66d9a3",
                 label="|e_ct| (PPO)")
    axes[2].plot(t_ppo, _col(ppo_rows, "wheel_utilization"), color="#ffce6b",
                 label="wheel utilisation")
    axes[2].set_ylabel("error (cm) / utilisation")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(fontsize=8)

    figure.suptitle(
        f"scenario {index}: {scenario['path_key']} @ {scenario['v_target']} m/s, "
        f"mass {scenario['mass']:.0f} kg, delay {scenario['actuator_delay_s']*1000:.0f} ms",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=120)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--from-confirmation", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--scenario-seed", type=int, default=4242)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=12)
    parser.add_argument("--plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = GainCalibration.load(args.calibration)
    model = PPO.load(args.model, device="cpu")
    arms = resolve_arms(args.model)
    outcomes = load_outcomes(args.from_confirmation)
    scenarios = sample_scenarios(args.episodes, args.scenario_seed, config.TRAIN_PATHS)

    fixed_names = [name for name in outcomes if name != "PPO"]
    decisive = [
        i for i in range(len(scenarios))
        if outcomes["PPO"].get(i) and not any(outcomes[n].get(i) for n in fixed_names)
    ]
    shared = [
        i for i in range(len(scenarios))
        if outcomes["PPO"].get(i) and all(outcomes[n].get(i) for n in fixed_names)
    ]
    print(f"{len(decisive)} DECISIVE scenarios (policy finishes, no fixed gain does)")
    print(f"{len(shared)} scenarios every controller finishes (control group)\n")
    if not decisive:
        raise SystemExit("no decisive scenarios; nothing to characterise")

    output_dir = args.output_dir or args.model.parent / "within_episode"
    output_dir.mkdir(parents=True, exist_ok=True)

    def analyse(indices: list[int], label: str, make_plots: bool = False):
        per_episode: list[dict[str, float]] = []
        lags: list[tuple[int, float]] = []
        for count, index in enumerate(indices[: args.max_episodes]):
            scenario = scenarios[index]
            _metrics, ppo_rows = replay(calibration, scenario, arms, model=model)
            extra = curvature_features(ppo_rows, scenario["path_key"])
            gain = _col(ppo_rows, "kp")
            row = {
                name: safe_corr(gain, func(ppo_rows, extra))
                for name, func in FEATURES.items()
            }
            row["_spread"] = float(gain.max() - gain.min())
            span = float(calibration.high[0] - calibration.low[0])
            near_rail = (gain <= calibration.low[0] + 0.02 * span) | (
                gain >= calibration.high[0] - 0.02 * span
            )
            row["_rail_fraction"] = float(np.mean(near_rail))
            row["_mean_step"] = float(np.mean(np.abs(np.diff(gain)))) if len(gain) > 1 else 0.0
            per_episode.append(row)
            lags.append(lead_lag(gain, np.abs(_col(ppo_rows, "e_ct"))))
            if make_plots:
                _m, fixed_rows = replay(
                    calibration, scenario, arms, gains=calibration.base
                )
                plot_episode(index, scenario, ppo_rows, fixed_rows, calibration,
                             output_dir / f"decisive_{index:03d}.png")
            print(f"  {label} {count + 1}/{min(len(indices), args.max_episodes)}",
                  end="\r", flush=True)
        print(" " * 40, end="\r")

        print(f"\n{label}: mean within-episode correlation of Kp against")
        print(f"{'feature':>20}{'corr':>9}{'seen in':>10}")
        summary: dict[str, float] = {}
        for name in FEATURES:
            values = np.array([r[name] for r in per_episode])
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            strong = int(np.sum(np.abs(values) > 0.3))
            summary[name] = float(values.mean())
            print(f"{name:>20}{values.mean():>9.2f}{strong:>6}/{len(values)}")
        spreads = np.array([r["_spread"] for r in per_episode])
        print(f"{'Kp spread in episode':>20}{spreads.mean():>9.1f}"
              f"   (box is {calibration.high[0] - calibration.low[0]:.0f} wide)")
        valid = [(l, v) for l, v in lags if np.isfinite(v)]
        if valid:
            mean_lag = float(np.mean([l for l, _ in valid]))
            mean_peak = float(np.mean([v for _, v in valid]))
            # The lag is only meaningful if the correlation it peaks at is
            # meaningful. Naming a direction off |corr| = 0.15 dresses noise up
            # as a finding: the argmax of a flat, noisy curve lands somewhere,
            # and that somewhere means nothing.
            if abs(mean_peak) < 0.3:
                print(f"  best Kp/|e_ct| alignment peaks at only {mean_peak:+.2f} "
                      f"(lag {mean_lag:+.1f} steps) -- too weak to call a direction; "
                      "the gain is not tracking the error at any lag")
            else:
                direction = ("REACTS to error" if mean_lag > 1 else
                             "ANTICIPATES error" if mean_lag < -1 else
                             "moves with error")
                print(f"  best Kp/|e_ct| alignment at lag {mean_lag:+.1f} control "
                      f"steps ({mean_lag * 0.02:+.2f} s), peak corr {mean_peak:+.2f}"
                      f" -> {direction}")
        # A policy dithering between the extremes and one following a smooth
        # rule can show the same spread. The fraction of time spent pinned at a
        # rail, and the step-to-step change, tell them apart.
        rails = np.array([r["_rail_fraction"] for r in per_episode])
        steps = np.array([r["_mean_step"] for r in per_episode])
        print(f"{'time at a rail':>20}{rails.mean():>9.0%}")
        print(f"{'mean |dKp| per step':>20}{steps.mean():>9.1f}"
              f"   (a smooth schedule would be a small fraction of the box)")
        return summary
        return summary

    decisive_summary = analyse(decisive, "decisive", make_plots=args.plots)
    control_summary = analyse(shared, "control") if shared else {}

    if control_summary:
        print("\nwhat is DIFFERENT about the decisive episodes")
        print(f"{'feature':>20}{'decisive':>10}{'control':>9}{'delta':>8}")
        for name in decisive_summary:
            if name not in control_summary:
                continue
            delta = decisive_summary[name] - control_summary[name]
            flag = "  <--" if abs(delta) > 0.15 else ""
            print(f"{name:>20}{decisive_summary[name]:>10.2f}"
                  f"{control_summary[name]:>9.2f}{delta:>8.2f}{flag}")

    report = output_dir / "within_episode.json"
    report.write_text(
        json.dumps(
            {
                "model": str(args.model),
                "decisive_scenarios": decisive,
                "decisive_correlations": decisive_summary,
                "control_correlations": control_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved {report}")
    if args.plots:
        print(f"Plots in {output_dir}")


if __name__ == "__main__":
    main()
