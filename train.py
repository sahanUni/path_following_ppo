"""Train the one-seed development PPO steering-gain scheduler."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from calibration import GainCalibration
from callbacks import CurriculumAndMetricsCallback, DevelopmentEvalCallback
import config
from env import PathFollowingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/development"))
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--n-steps", type=int, default=config.PPO_KWARGS["n_steps"])
    parser.add_argument(
        "--no-gusts",
        action="store_true",
        help=(
            "Drop wind gusts from the stage-3 disturbance mix. This is the "
            "control run for the gust study: identical in every other respect."
        ),
    )
    parser.add_argument(
        "--no-delay",
        action="store_true",
        help=(
            "Train with no actuator dead time. Ablation control: dead time is "
            "what puts a stability ceiling on Kp, so without it the best gain "
            "is again whatever saturates the wheels first."
        ),
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help=(
            "Torch intra-op threads. Defaults to 1 because the normal way to "
            "use a many-core machine here is one process per seed, and torch "
            "otherwise grabs every core in EVERY process -- five runs each "
            "spawning 32 threads on 32 cores is slower than five single-threaded "
            "ones. Raise it only when running a single job alone."
        ),
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help=(
            "Train with a perfect cross-track sensor. Ablation control: noise "
            "is what makes Kd a real decision rather than free information."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Set before the first tensor op. Thread count changes floating-point
    # reduction order, so two runs that differ only in this can diverge -- fix
    # it explicitly wherever runs are meant to be comparable rather than
    # inheriting whatever the machine defaults to.
    torch.set_num_threads(max(1, args.torch_threads))
    if not args.calibration.exists():
        raise SystemExit(
            f"Calibration not found: {args.calibration}\n"
            "Run calibrate_pid.py before PPO training."
        )
    calibration = GainCalibration.load(args.calibration)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    gusts = not args.no_gusts
    delay = not args.no_delay
    noise = not args.no_noise
    print(f"Stage-3 disturbances: mass, force{', gust' if gusts else ' (gusts DISABLED)'}")
    print(
        "Plant imperfections: "
        f"dead time {'ON' if delay else 'DISABLED'}, "
        f"sensor noise {'ON' if noise else 'DISABLED'}"
    )

    env_kwargs = {
        "calibration": calibration,
        "training": True,
        "gusts": gusts,
        "delay": delay,
        "noise": noise,
    }

    validation_env = PathFollowingEnv(**env_kwargs)
    check_env(validation_env, warn=True)
    validation_env.close()

    monitor_path = args.run_dir / "monitor.csv"

    def make_env() -> Monitor:
        return Monitor(PathFollowingEnv(**env_kwargs), filename=str(monitor_path))

    vec_env = DummyVecEnv([make_env])
    ppo_kwargs = dict(config.PPO_KWARGS)
    ppo_kwargs["n_steps"] = args.n_steps
    if ppo_kwargs["batch_size"] > args.n_steps:
        ppo_kwargs["batch_size"] = args.n_steps
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(args.run_dir / "tensorboard"),
        device="cpu",
        **ppo_kwargs,
    )
    callbacks = CallbackList(
        [
            CurriculumAndMetricsCallback(
                args.total_timesteps, args.run_dir / "episodes.csv"
            ),
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq, 1),
                save_path=str(args.run_dir / "checkpoints"),
                name_prefix="ppo_path_following",
            ),
            DevelopmentEvalCallback(
                calibration=calibration,
                eval_freq=args.eval_freq,
                output_dir=args.run_dir,
                seed=args.seed,
            ),
        ]
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        tb_log_name="PPO",
    )
    model.save(args.run_dir / "final_model")
    vec_env.close()
    print(f"Saved final model to {args.run_dir / 'final_model.zip'}")


if __name__ == "__main__":
    main()

