"""Typed, serializable scenarios for paired adaptation evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterable

import config
from core import paths


EVENT_KINDS = {"none", "mass_step", "force_pulse"}
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,119}$")


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class PhysicsConfig:
    mass: float = config.NOMINAL_MASS_KG
    friction_scale: float = config.NOMINAL_FRICTION
    actuator_scale: float = config.NOMINAL_ACTUATOR

    def __post_init__(self) -> None:
        mass = _finite(self.mass, "mass")
        friction = _finite(self.friction_scale, "friction_scale")
        actuator = _finite(self.actuator_scale, "actuator_scale")
        if not config.MASS_RANGE_KG[0] <= mass <= config.MASS_RANGE_KG[1]:
            raise ValueError(f"mass must be within {config.MASS_RANGE_KG}")
        if not config.FRICTION_RANGE[0] <= friction <= config.FRICTION_RANGE[1]:
            raise ValueError(f"friction_scale must be within {config.FRICTION_RANGE}")
        if not config.ACTUATOR_RANGE[0] <= actuator <= config.ACTUATOR_RANGE[1]:
            raise ValueError(f"actuator_scale must be within {config.ACTUATOR_RANGE}")
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "friction_scale", friction)
        object.__setattr__(self, "actuator_scale", actuator)


@dataclass(frozen=True)
class EventDefinition:
    kind: str = "none"
    start_time_s: float = 0.0
    end_time_s: float | None = None
    value: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"event kind must be one of {sorted(EVENT_KINDS)}")
        start = _finite(self.start_time_s, "start_time_s")
        value = _finite(self.value, "event value")
        end = None if self.end_time_s is None else _finite(self.end_time_s, "end_time_s")
        if start < 0.0:
            raise ValueError("start_time_s cannot be negative")
        if end is not None and end <= start:
            raise ValueError("end_time_s must be later than start_time_s")
        if self.kind == "none":
            if start != 0.0 or end is not None or value != 0.0:
                raise ValueError("none events must use zero start/value and null end")
        elif self.kind == "mass_step":
            if end is not None:
                raise ValueError("mass_step is persistent and must have a null end")
            if not config.MASS_RANGE_KG[0] <= value <= config.MASS_RANGE_KG[1]:
                raise ValueError(f"mass-step target must be within {config.MASS_RANGE_KG}")
        elif end is None:
            raise ValueError("force_pulse requires an end_time_s")
        object.__setattr__(self, "start_time_s", start)
        object.__setattr__(self, "end_time_s", end)
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    path_key: str
    target_speed: float
    seed: int
    base_physics: PhysicsConfig
    event: EventDefinition = EventDefinition()
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.scenario_id):
            raise ValueError("scenario_id must be a lowercase filesystem-safe identifier")
        if self.path_key not in paths.CATALOGUE:
            raise ValueError(f"unknown path key: {self.path_key}")
        speed = _finite(self.target_speed, "target_speed")
        if speed not in config.SPEED_TARGETS:
            raise ValueError(f"target_speed must be one of {config.SPEED_TARGETS}")
        seed = int(self.seed)
        if seed < 0:
            raise ValueError("seed cannot be negative")
        tags = tuple(str(tag).strip().lower() for tag in self.tags)
        if any(not _ID_PATTERN.fullmatch(tag) for tag in tags):
            raise ValueError("tags must be lowercase filesystem-safe identifiers")
        object.__setattr__(self, "target_speed", speed)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "tags", tags)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tags"] = list(self.tags)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Scenario":
        allowed = {"scenario_id", "path_key", "target_speed", "seed", "base_physics", "event", "tags"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown scenario fields: {sorted(unknown)}")
        return cls(
            scenario_id=payload["scenario_id"],
            path_key=payload["path_key"],
            target_speed=payload["target_speed"],
            seed=payload["seed"],
            base_physics=PhysicsConfig(**payload["base_physics"]),
            event=EventDefinition(**payload.get("event", {})),
            tags=tuple(payload.get("tags", ())),
        )

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def to_env_options(self) -> dict[str, Any]:
        event = self.event
        if event.kind == "none":
            disturbance = {"kind": "none", "start_s": 0.0, "end_s": None, "amount": 0.0}
        elif event.kind == "mass_step":
            disturbance = {
                "kind": "mass_step",
                "start_s": event.start_time_s,
                "end_s": None,
                "amount": event.value,
            }
        else:
            disturbance = {
                "kind": "force_pulse",
                "start_s": event.start_time_s,
                "end_s": event.end_time_s,
                "amount": event.value,
            }
        return {
            "path_key": self.path_key,
            "v_target": self.target_speed,
            "mass": self.base_physics.mass,
            "friction": self.base_physics.friction_scale,
            "actuator": self.base_physics.actuator_scale,
            "stage": 3,
            "disturbance": disturbance,
        }


def _number_token(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def _event_start(path_key: str, speed: float) -> float:
    ideal_time = float(paths.get(path_key)["length"]) / float(speed)
    return round(config.ADAPTATION_EVENT_START_FRACTION * ideal_time, 3)


def _scenario(
    path_key: str,
    speed: float,
    *,
    seed: int,
    physics: PhysicsConfig,
    event: EventDefinition,
    label: str,
    tags: tuple[str, ...],
) -> Scenario:
    scenario_id = f"{path_key}_v{_number_token(speed)}_{label}"
    return Scenario(scenario_id, path_key, speed, seed, physics, event, tags)


def build_scenarios(
    path_keys: Iterable[str],
    speeds: Iterable[float],
    *,
    seed: int = config.SEED,
) -> list[Scenario]:
    """Build the explicit one-axis-at-a-time first-version matrix."""
    scenarios: list[Scenario] = []
    nominal = PhysicsConfig()
    for path_key in path_keys:
        for speed in speeds:
            start = _event_start(path_key, speed)
            base_tags = ("development",)
            scenarios.append(_scenario(path_key, speed, seed=seed, physics=nominal,
                event=EventDefinition(), label="nominal", tags=base_tags + ("nominal",)))
            for mass in config.ADAPTATION_MASS_LEVELS_KG:
                if mass == config.NOMINAL_MASS_KG:
                    continue
                scenarios.append(_scenario(path_key, speed, seed=seed,
                    physics=PhysicsConfig(mass=mass), event=EventDefinition(),
                    label=f"mass_{_number_token(mass)}", tags=base_tags + ("stationary", "mass")))
            for friction in config.ADAPTATION_FRICTION_LEVELS:
                if friction == config.NOMINAL_FRICTION:
                    continue
                scenarios.append(_scenario(path_key, speed, seed=seed,
                    physics=PhysicsConfig(friction_scale=friction), event=EventDefinition(),
                    label=f"friction_{_number_token(friction)}", tags=base_tags + ("stationary", "friction")))
            for actuator in config.ADAPTATION_ACTUATOR_LEVELS:
                if actuator == config.NOMINAL_ACTUATOR:
                    continue
                scenarios.append(_scenario(path_key, speed, seed=seed,
                    physics=PhysicsConfig(actuator_scale=actuator), event=EventDefinition(),
                    label=f"actuator_{_number_token(actuator)}", tags=base_tags + ("stationary", "actuator")))
            for before, after in config.ADAPTATION_MASS_STEPS_KG:
                scenarios.append(_scenario(path_key, speed, seed=seed,
                    physics=PhysicsConfig(mass=before),
                    event=EventDefinition("mass_step", start, None, after),
                    label=f"mass_{_number_token(before)}_to_{_number_token(after)}",
                    tags=base_tags + ("dynamic", "mass")))
            for force in config.ADAPTATION_FORCE_PULSES_N:
                scenarios.append(_scenario(path_key, speed, seed=seed, physics=nominal,
                    event=EventDefinition("force_pulse", start,
                        start + config.ADAPTATION_FORCE_DURATION_S, force),
                    label=f"force_{_number_token(force)}",
                    tags=base_tags + ("dynamic", "force")))
    validate_scenario_set(scenarios)
    return scenarios


def validate_scenario_set(scenarios: Iterable[Scenario]) -> None:
    seen_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    for scenario in scenarios:
        if scenario.scenario_id in seen_ids:
            raise ValueError(f"duplicate scenario_id: {scenario.scenario_id}")
        if scenario.fingerprint in seen_fingerprints:
            raise ValueError(f"duplicate scenario definition: {scenario.scenario_id}")
        seen_ids.add(scenario.scenario_id)
        seen_fingerprints.add(scenario.fingerprint)


def smoke_scenarios(seed: int = config.SEED) -> list[Scenario]:
    """A trace-inspectable subset covering nominal and both event types."""
    full = build_scenarios(("slalom",), (0.5,), seed=seed)
    suffixes = ("_nominal", "_mass_10_to_30", "_force_neg5", "_force_5")
    return [scenario for scenario in full if scenario.scenario_id.endswith(suffixes)]
