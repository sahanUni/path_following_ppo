"""SB3 callbacks for curriculum, auditable episode logs, and fixed evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from calibration import GainCalibration
import config
from env import PathFollowingEnv
from rollout import run_episode


def evaluation_conditions(
    seed: int,
) -> tuple[tuple[str, dict[str, Any], list[dict[str, Any]]], ...]:
    """The three evaluation tiers, as (name, plant overrides, disturbances).

    Calm and gust both run a nominal car that completes at every reachable
    gain, so on their own they score tracking and nothing else -- and tracking
    is not the axis the gain mostly controls. `stress` is where FINISHING is in
    doubt and depends on the gain; see config.EVAL_STRESS_* for the numbers
    behind it. Without that tier the criterion is blind to the only axis that
    improved, which is why run v7_seed7 sat flat for all 40 checkpoints and
    froze `best_model` at 50k steps.
    """
    no_event = [{"kind": "none", "start_s": 0.0, "end_s": None, "amount": 0.0}]
    return (
        ("calm", {}, no_event),
        (
            "gust",
            {},
            [
                {
                    "kind": "force_gust",
                    "start_s": 0.0,
                    "end_s": None,
                    "amount": config.EVAL_GUST_SIGMA_N,
                    "tau_s": config.EVAL_GUST_TAU_S,
                    "seed": seed,
                }
            ],
        ),
        (
            "stress",
            {
                "mass": config.EVAL_STRESS_MASS_KG,
                "friction": config.EVAL_STRESS_FRICTION,
                "actuator_delay_s": config.EVAL_STRESS_ACTUATOR_DELAY_S,
                "sensor_noise_m": config.EVAL_STRESS_SENSOR_NOISE_M,
            },
            no_event,
        ),
    )


class CurriculumAndMetricsCallback(BaseCallback):
    def __init__(self, total_timesteps: int, output_csv: Path) -> None:
        super().__init__(verbose=0)
        self.total_timesteps = max(int(total_timesteps), 1)
        self.output_csv = output_csv
        self._handle = None
        self._writer = None

    def _on_training_start(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.output_csv.open("w", newline="", encoding="utf-8")

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_timesteps)
        self.training_env.env_method("set_training_progress", progress)
        for info in self.locals.get("infos", []):
            metrics = info.get("episode_metrics")
            if not metrics:
                continue
            row = {"timesteps": self.num_timesteps, **metrics}
            if self._writer is None:
                self._writer = csv.DictWriter(self._handle, fieldnames=list(row))
                self._writer.writeheader()
            self._writer.writerow(row)
            self._handle.flush()
            for key, value in metrics.items():
                if isinstance(value, (int, float, bool)):
                    self.logger.record(f"episode/{key}", float(value))
        return True

    def _on_training_end(self) -> None:
        if self._handle is not None:
            self._handle.close()


class DevelopmentEvalCallback(BaseCallback):
    """Evaluate by completion first and tracking distance second."""

    def __init__(
        self,
        calibration: GainCalibration,
        eval_freq: int,
        output_dir: Path,
        seed: int,
    ) -> None:
        super().__init__(verbose=1)
        self.calibration = calibration
        self.eval_freq = max(int(eval_freq), 1)
        self.output_dir = output_dir
        self.seed = int(seed)
        self._last_eval = 0
        self._best_score: tuple[int, float] | None = None
        self._history_path = output_dir / "evaluation_history.csv"
        self._history_handle = None
        self._history_writer = None

    def _on_training_start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._history_handle = self._history_path.open("w", newline="", encoding="utf-8")

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps
        # Chosen for sensitivity: Kp 26 vs Kp 50 separates these by 22-34% under
        # the eval gust, at every speed, with all episodes still finishing. The
        # previous set led with `arc`, which separates by 0.4% at 0.3 m/s and so
        # contributed almost nothing but averaging weight to the score.
        path_keys = ("hairpin", "square", "zigzag30")
        env = PathFollowingEnv(
            calibration=self.calibration,
            training=False,
            path_keys=path_keys,
        )

        def policy(observation: np.ndarray) -> np.ndarray:
            action, _ = self.model.predict(observation, deterministic=True)
            return np.asarray(action)

        conditions = evaluation_conditions(self.seed)

        rows: list[dict[str, Any]] = []
        by_condition: dict[str, list[dict[str, Any]]] = {
            name: [] for name, _, _ in conditions
        }
        for path_key in path_keys:
            for speed in config.SPEED_TARGETS:
                for name, overrides, disturbances in conditions:
                    metrics, _ = run_episode(
                        env,
                        policy,
                        seed=self.seed,
                        options={
                            "path_key": path_key,
                            "v_target": speed,
                            "mass": config.NOMINAL_MASS_KG,
                            "friction": config.NOMINAL_FRICTION,
                            "actuator": config.NOMINAL_ACTUATOR,
                            # Fixed, not nominal: a checkpoint scored on a plant
                            # with no dead time and no sensor noise is scored on
                            # the plant where high gain is free, so selection
                            # would reward exactly the constant-max-Kp policy
                            # the study is trying to move away from. Held
                            # constant so scores stay comparable across
                            # checkpoints and runs.
                            "actuator_delay_s": config.EVAL_ACTUATOR_DELAY_S,
                            "sensor_noise_m": config.EVAL_SENSOR_NOISE_M,
                            "noise_seed": self.seed,
                            "stage": 3,
                            "disturbances": disturbances,
                            **overrides,
                        },
                    )
                    rows.append(metrics)
                    by_condition[name].append(metrics)
        env.close()
        completed = sum(int(row["finished"]) for row in rows)
        # Averaged over FINISHED episodes only. An episode that leaves the
        # corridor lands 40x the error of one that completes, so a single
        # unwinnable condition otherwise supplies most of the score and drowns
        # the differences among the episodes that actually respond to the gains.
        # Failures are not ignored -- `completed` is the primary sort key below,
        # so failing more episodes is still strictly worse.
        finished = [row["mean_distance_m"] for row in rows if row["finished"]]
        mean_distance = float(np.mean(finished)) if finished else float("inf")
        row = {
            "timesteps": self.num_timesteps,
            "completed": completed,
            "episodes": len(rows),
            "completion_rate": completed / len(rows),
            "mean_distance_m": mean_distance,
            "mean_distance_all_m": float(
                np.mean([item["mean_distance_m"] for item in rows])
            ),
            "mean_itae": float(np.mean([x["itae"] for x in rows])),
            "mean_saturation_fraction": float(
                np.mean([x["saturation_fraction"] for x in rows])
            ),
        }
        # Broken out so a gust-driven gain is visible in the history rather than
        # being averaged away against the (much smaller) calm errors.
        for name, group in by_condition.items():
            done = [item["mean_distance_m"] for item in group if item["finished"]]
            row[f"mean_distance_{name}_m"] = float(np.mean(done)) if done else float("inf")
            row[f"completed_{name}"] = sum(int(item["finished"]) for item in group)
        if self._history_writer is None:
            self._history_writer = csv.DictWriter(
                self._history_handle, fieldnames=list(row)
            )
            self._history_writer.writeheader()
        self._history_writer.writerow(row)
        self._history_handle.flush()
        for key, value in row.items():
            if key != "timesteps":
                self.logger.record(f"eval/{key}", value)

        score = (completed, -mean_distance)
        if self._best_score is None or score > self._best_score:
            self._best_score = score
            self.model.save(self.output_dir / "best_model")
            if self.verbose:
                print(
                    f"New best development model: {completed}/{len(rows)} complete, "
                    f"mean distance {mean_distance:.4f} m"
                )
        return True

    def _on_training_end(self) -> None:
        if self._history_handle is not None:
            self._history_handle.close()

