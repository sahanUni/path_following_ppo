"""Gain calibration artifact and the PPO action-to-gain mapping."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GainCalibration:
    """One global fixed-PID baseline plus the legal PPO gain box."""

    base: np.ndarray
    low: np.ndarray
    high: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        base = np.asarray(self.base, dtype=np.float64)
        low = np.asarray(self.low, dtype=np.float64)
        high = np.asarray(self.high, dtype=np.float64)
        if base.shape != (3,) or low.shape != (3,) or high.shape != (3,):
            raise ValueError("base, low and high must each contain [Kp, Ki, Kd]")
        if np.any(low < 0.0) or np.any(high <= low):
            raise ValueError("gain bounds must be non-negative and strictly ordered")
        if np.any(base < low) or np.any(base > high):
            raise ValueError("baseline gains must lie inside the gain bounds")
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @classmethod
    def development_default(cls) -> "GainCalibration":
        """Test-only fallback; train.py intentionally requires a saved artifact."""
        return cls(
            base=np.array([8.0, 0.0, 0.3]),
            low=np.array([0.5, 0.0, 0.0]),
            high=np.array([25.0, 2.0, 3.0]),
            metadata={"status": "uncalibrated-development-default"},
        )

    @classmethod
    def load(cls, path: str | Path) -> "GainCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            base=np.asarray(payload["base"], dtype=np.float64),
            low=np.asarray(payload["low"], dtype=np.float64),
            high=np.asarray(payload["high"], dtype=np.float64),
            metadata=dict(payload.get("metadata", {})),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "base": self.base.tolist(),
            "low": self.low.tolist(),
            "high": self.high.tolist(),
            "metadata": self.metadata,
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def gains_from_action(self, action: np.ndarray) -> np.ndarray:
        """Piecewise map: -1 -> low, 0 -> baseline, +1 -> high."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        below = self.base + action * (self.base - self.low)
        above = self.base + action * (self.high - self.base)
        return np.where(action < 0.0, below, above)

