"""Evaluate a PPO model or the calibrated fixed PID on identical scenarios."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from calibration import GainCalibration
import config
from env import PathFollowingEnv
from rollout import fixed_policy, run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=Path)
    group.add_argument("--fixed-pid", action="store_true")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--path-set",
        choices=("training", "held-out", "stress", "all"),
        default="training",
    )
    parser.add_argument("--paths", nargs="+")
    parser.add_argument("--speeds", nargs="+", type=float, default=list(config.SPEED_TARGETS))
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/evaluation"))
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def selected_paths(args: argparse.Namespace) -> tuple[str, ...]:
    if args.paths:
        return tuple(args.paths)
    groups = {
        "training": config.TRAIN_PATHS,
        "held-out": config.HELD_OUT_PATHS,
        "stress": config.STRESS_PATHS,
        "all": config.TRAIN_PATHS + config.HELD_OUT_PATHS + config.STRESS_PATHS,
    }
    return groups[args.path_set]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    calibration = GainCalibration.load(args.calibration)
    path_keys = selected_paths(args)
    fixed = bool(args.fixed_pid)
    env = PathFollowingEnv(
        calibration=calibration,
        training=False,
        path_keys=path_keys,
        fixed_gains=calibration.base if fixed else None,
    )
    if fixed:
        policy = fixed_policy
        label = "fixed_pid"
    else:
        model = PPO.load(args.model, device="cpu")

        def policy(observation: np.ndarray) -> np.ndarray:
            action, _ = model.predict(observation, deterministic=True)
            return np.asarray(action)

        label = "ppo"

    summaries: list[dict] = []
    traces: list[dict] = []
    episode_id = 0
    for path_key in path_keys:
        for speed in args.speeds:
            episode_id += 1
            metrics, trace = run_episode(
                env,
                policy,
                seed=args.seed,
                options={
                    "path_key": path_key,
                    "v_target": speed,
                    "mass": config.NOMINAL_MASS_KG,
                    "friction": config.NOMINAL_FRICTION,
                    "actuator": config.NOMINAL_ACTUATOR,
                    "stage": 2,
                },
            )
            summaries.append({"controller": label, "episode_id": episode_id, **metrics})
            traces.extend(
                {"controller": label, "episode_id": episode_id, **row}
                for row in trace
            )
            print(
                f"{path_key:>10} v*={speed:.1f}: "
                f"finished={metrics['finished']} "
                f"mean_dist={metrics['mean_distance_m']:.4f} m "
                f"sat={100 * metrics['saturation_fraction']:.1f}%"
            )
    env.close()
    summary_path = args.output_dir / f"{label}_summary.csv"
    trace_path = args.output_dir / f"{label}_traces.csv"
    write_csv(summary_path, summaries)
    write_csv(trace_path, traces)
    print(f"Saved {summary_path} and {trace_path}")
    if args.plot:
        from plot_results import plot_trace_file

        plot_trace_file(trace_path, args.output_dir / f"{label}_episodes.png")


if __name__ == "__main__":
    main()

