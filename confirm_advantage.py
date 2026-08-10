"""Paired PPO-versus-fixed-PID comparison on sampled training scenarios.

Answers one question with a defensible number: does the trained policy beat the
fixed baseline, or did it draw an easier set of episodes?

Every controller meets the SAME scenarios with the same seeds -- same path,
speed, mass, friction, dead time, sensor-noise trace and disturbance -- so the
comparison is paired and scenario difficulty cancels out. What remains is
whether the wins repeat, which is what the episode count buys.

Scenarios are SAMPLED from the stage-3 training distribution rather than chosen
by hand. Hand-picked conditions are readable precisely because everything
completes on them, and on those the gain only moves tracking by a fraction of a
millimetre; the axis it actually controls is whether the episode finishes.
"""

from __future__ import annotations

import argparse
import csv
import json
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from calibration import GainCalibration
import config
from env import PathFollowingEnv
from rollout import fixed_policy, run_episode


def resolve_arms(model_path: Path, override: dict[str, bool] | None = None) -> dict[str, bool]:
    """Which observation arms the model was trained with.

    Read from the `arms.json` train.py writes beside the model, because the
    observation width cannot identify them on its own: preview and plant
    context add five values each, so 175 is ambiguous between the two. Getting
    this wrong produces an opaque shape error deep inside SB3, so it is
    resolved explicitly and printed.
    """
    if override is not None:
        return override
    manifest = model_path.parent / "arms.json"
    if manifest.exists():
        loaded = json.loads(manifest.read_text(encoding="utf-8"))
        return {
            "preview": bool(loaded.get("preview", False)),
            "plant_context": bool(loaded.get("plant_context", False)),
        }
    return {"preview": False, "plant_context": False}


def sample_scenarios(count: int, seed: int, paths: tuple[str, ...]) -> list[dict[str, Any]]:
    """Draw stage-3 episodes from the training distribution, policy-free.

    Resolved to concrete values here, once, so that every controller replays an
    identical scenario instead of re-drawing its own.
    """
    env = PathFollowingEnv(training=True, path_keys=paths)
    scenarios: list[dict[str, Any]] = []
    try:
        for index in range(count):
            env.reset(seed=seed + index, options={"stage": 3})
            scenarios.append(
                {
                    "path_key": env.path_key,
                    "v_target": env.v_target,
                    "mass": env.mass,
                    "friction": env.friction,
                    "actuator": env.actuator,
                    "actuator_delay_s": env.actuator_delay_s,
                    "sensor_noise_m": env.sensor_noise_m,
                    "noise_seed": env.noise_seed,
                    "stage": 3,
                    "disturbances": [dict(env.disturbance)],
                }
            )
    finally:
        env.close()
    return scenarios


def run_one(
    calibration: GainCalibration,
    options: dict[str, Any],
    model: PPO | None = None,
    gains: np.ndarray | None = None,
    arms: dict[str, bool] | None = None,
) -> dict[str, Any]:
    arms = arms or {"preview": False, "plant_context": False}
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
        metrics, trace = run_episode(
            env,
            fixed_policy if gains is not None else ppo_policy,
            seed=config.SEED,
            options=options,
        )
    finally:
        env.close()
    metrics["mean_kp"] = float(np.mean([row["kp"] for row in trace]))
    return metrics


def exact_mcnemar(wins: int, losses: int) -> float:
    """Two-sided exact p-value for a paired win/loss count.

    Only episodes where the two controllers DISAGREE carry information: the
    ones both finish, or both fail, say nothing about which is better. Under
    the null the disagreements split like a fair coin, so this is the
    probability of a split at least this lopsided.
    """
    total = wins + losses
    if total == 0:
        return 1.0
    extreme = min(wins, losses)
    tail = sum(comb(total, k) for k in range(extreme + 1)) / (2.0**total)
    return min(1.0, 2.0 * tail)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help=(
            "Sampled scenarios per controller. At 60 the random wobble in a "
            "completion count is around +/-5 episodes, which swamps the "
            "difference being measured; 200 brings it to about +/-2.5."
        ),
    )
    parser.add_argument(
        "--scenario-seed",
        type=int,
        default=4242,
        help="Held apart from config.SEED so the suite is not the training draw.",
    )
    parser.add_argument(
        "--extra-kp",
        type=str,
        default="",
        help=(
            "Comma-separated additional fixed Kp values to include as "
            "reference controllers, e.g. 15,20,50. The calibrated baseline is "
            "always included."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = GainCalibration.load(args.calibration)
    model = PPO.load(args.model, device="cpu")
    arms = resolve_arms(args.model)
    expected = model.observation_space.shape[0]
    probe = PathFollowingEnv(
        calibration=calibration, training=False,
        preview=arms["preview"], plant_context=arms["plant_context"],
    )
    actual = probe.observation_space.shape[0]
    probe.close()
    if actual != expected:
        raise SystemExit(
            f"observation mismatch: the model expects {expected} values, the "
            f"environment with arms {arms} produces {actual}. Check "
            f"{args.model.parent / 'arms.json'}."
        )
    scenarios = sample_scenarios(args.episodes, args.scenario_seed, config.TRAIN_PATHS)
    print(
        f"{len(scenarios)} scenarios sampled from the stage-3 training "
        f"distribution (seed {args.scenario_seed})"
    )
    print(f"model:       {args.model}")
    print(f"calibration: {args.calibration}  base={calibration.base.round(3).tolist()}")
    print(f"arms:        {arms}  (observation width {expected})\n")

    controllers: list[tuple[str, np.ndarray | None]] = [
        ("PPO", None),
        (f"fixed Kp {calibration.base[0]:.0f} (baseline)", calibration.base.copy()),
    ]
    for token in [t for t in args.extra_kp.replace(",", " ").split() if t]:
        gains = calibration.base.copy()
        gains[0] = float(token)
        controllers.append((f"fixed Kp {float(token):.0f}", gains))

    results: dict[str, list[dict[str, Any]]] = {}
    for name, gains in controllers:
        outcomes = []
        for index, scenario in enumerate(scenarios, start=1):
            outcomes.append(
                run_one(calibration, scenario, model=model, gains=gains, arms=arms)
            )
            if index % 50 == 0:
                print(f"  {name}: {index}/{len(scenarios)}", flush=True)
        results[name] = outcomes
        done = sum(int(o["finished"]) for o in outcomes)
        print(f"  {name}: {done}/{len(scenarios)} complete", flush=True)

    print(f"\n{'controller':>28}{'completed':>12}{'rate':>8}{'mean cm':>10}{'mean Kp':>10}")
    for name, outcomes in results.items():
        done = [o for o in outcomes if o["finished"]]
        mean = np.mean([o["mean_distance_m"] for o in done]) * 100 if done else float("nan")
        kp = np.mean([o["mean_kp"] for o in outcomes])
        print(
            f"{name:>28}{len(done):>8}/{len(outcomes):<3}"
            f"{len(done)/len(outcomes):8.3f}{mean:10.2f}{kp:10.1f}"
        )

    # Paired completion tests, PPO against each fixed reference.
    print(f"\n{'paired vs PPO':>28}{'PPO only':>10}{'fixed only':>12}{'p (exact)':>12}")
    ppo_outcomes = results["PPO"]
    for name, outcomes in results.items():
        if name == "PPO":
            continue
        wins = sum(
            1
            for a, b in zip(ppo_outcomes, outcomes)
            if a["finished"] and not b["finished"]
        )
        losses = sum(
            1
            for a, b in zip(ppo_outcomes, outcomes)
            if b["finished"] and not a["finished"]
        )
        p_value = exact_mcnemar(wins, losses)
        verdict = "significant" if p_value < 0.05 else "not significant"
        print(f"{name:>28}{wins:>10}{losses:>12}{p_value:12.4f}   {verdict}")

    # The strongest fixed-gain comparison available: allow a DIFFERENT fixed
    # gain per scenario, chosen with hindsight. Beating this cannot be done by
    # any fixed gain, only by varying within the episode.
    fixed_names = [name for name in results if name != "PPO"]
    if len(fixed_names) > 1:
        oracle = sum(
            1
            for index in range(len(scenarios))
            if any(results[name][index]["finished"] for name in fixed_names)
        )
        ppo_done = sum(int(o["finished"]) for o in ppo_outcomes)
        print(
            f"\n  per-scenario fixed-gain ORACLE: {oracle}/{len(scenarios)}"
            f"   PPO: {ppo_done}/{len(scenarios)}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["scenario", "path_key", "v_target", "mass", "friction",
                 "actuator_delay_s", "sensor_noise_m", "event",
                 "controller", "finished", "mean_distance_m", "mean_kp"]
            )
            for index, scenario in enumerate(scenarios):
                for name, outcomes in results.items():
                    outcome = outcomes[index]
                    writer.writerow(
                        [index, scenario["path_key"], scenario["v_target"],
                         f"{scenario['mass']:.3f}", f"{scenario['friction']:.3f}",
                         f"{scenario['actuator_delay_s']:.4f}",
                         f"{scenario['sensor_noise_m']:.5f}",
                         scenario["disturbances"][0]["kind"], name,
                         int(outcome["finished"]),
                         f"{outcome['mean_distance_m']:.6f}",
                         f"{outcome['mean_kp']:.3f}"]
                    )
        print(f"\nSaved per-scenario results to {args.output}")


if __name__ == "__main__":
    main()
