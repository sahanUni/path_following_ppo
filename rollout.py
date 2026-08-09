"""Shared deterministic rollout helpers for evaluation and callbacks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from env import PathFollowingEnv


Policy = Callable[[np.ndarray], np.ndarray]


def run_episode(
    env: PathFollowingEnv,
    policy: Policy,
    *,
    seed: int,
    options: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observation, _ = env.reset(seed=seed, options=options)
    terminated = truncated = False
    total_reward = 0.0
    info: dict[str, Any] = {}
    while not (terminated or truncated):
        action = np.asarray(policy(observation), dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    metrics = dict(info["episode_metrics"])
    metrics["episode_reward"] = total_reward
    return metrics, env.get_trace()


def fixed_policy(_: np.ndarray) -> np.ndarray:
    return np.zeros(3, dtype=np.float32)

