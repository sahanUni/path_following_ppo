"""Shared trace-derived metrics for adaptation and paired reporting."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

import config
from adaptation_scenarios import Scenario


def _array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def _finite_values(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _safe_stat(values: np.ndarray, function) -> float | None:
    finite = _finite_values(values)
    return None if finite.size == 0 else float(function(finite))


def _integral(time_s: np.ndarray, values: np.ndarray) -> float | None:
    valid = np.isfinite(time_s) & np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        return None
    return float(np.trapezoid(values[valid], time_s[valid]))


def _window(time_s: np.ndarray, start: float, end: float | None) -> np.ndarray:
    mask = time_s >= start
    if end is not None:
        mask &= time_s < end
    return mask


def _first_sustained_recovery(
    time_s: np.ndarray,
    error: np.ndarray,
    *,
    search_start_s: float,
    threshold_m: float,
    sustained_s: float,
) -> float | None:
    valid = np.isfinite(time_s) & np.isfinite(error) & (time_s >= search_start_s)
    indices = np.flatnonzero(valid)
    run_start: float | None = None
    previous_time: float | None = None
    typical_dt = float(np.nanmedian(np.diff(time_s))) if time_s.size > 1 else 0.02
    for index in indices:
        current_time = float(time_s[index])
        contiguous = previous_time is None or current_time - previous_time <= 1.5 * typical_dt
        if error[index] <= threshold_m and contiguous:
            run_start = current_time if run_start is None else run_start
            if current_time - run_start + typical_dt >= sustained_s:
                return run_start
        elif error[index] <= threshold_m:
            run_start = current_time
        else:
            run_start = None
        previous_time = current_time
    return None


def adaptation_metrics(
    rows: Iterable[dict[str, Any]],
    scenario: Scenario,
) -> dict[str, Any]:
    """Calculate all dynamic metrics from a saved-compatible trace."""
    trace = list(rows)
    event = scenario.event
    if event.kind == "none" or not trace:
        return {
            "dynamic_event": False,
            "pre_median_error_m": None,
            "pre_peak_error_m": None,
            "post_peak_error_m": None,
            "post_event_iae": None,
            "post_pre_error_ratio": None,
            "recovery_threshold_m": None,
            "recovery_time_s": None,
            "failed_to_recover": None,
            "gain_response_time_s": None,
            "post_event_gain_change": None,
            "post_event_gain_chatter": None,
        }

    time_s = _array(trace, "t")
    error = np.abs(_array(trace, "dist"))
    start = event.start_time_s
    pre_mask = _window(time_s, max(0.0, start - config.ADAPTATION_PRE_WINDOW_S), start)
    response_end = start + config.ADAPTATION_RESPONSE_WINDOW_S
    response_mask = _window(time_s, start, response_end)
    post_mask = time_s >= start
    pre_median = _safe_stat(error[pre_mask], np.median)
    pre_peak = _safe_stat(error[pre_mask], np.max)
    post_peak = _safe_stat(error[response_mask], np.max)
    post_iae = _integral(time_s[post_mask], error[post_mask])
    post_median = _safe_stat(error[post_mask], np.median)
    ratio = None
    if pre_median is not None and post_median is not None:
        ratio = post_median / max(pre_median, 1e-12)

    threshold = None
    recovered_at = None
    recovery_time = None
    if pre_median is not None:
        threshold = (
            pre_median * (1.0 + config.ADAPTATION_RECOVERY_RELATIVE_TOLERANCE)
            + config.ADAPTATION_RECOVERY_FLOOR_M
        )
        recovery_search = max(response_end, event.end_time_s or start)
        recovered_at = _first_sustained_recovery(
            time_s,
            error,
            search_start_s=recovery_search,
            threshold_m=threshold,
            sustained_s=config.ADAPTATION_RECOVERY_SUSTAINED_S,
        )
        if recovered_at is not None:
            recovery_time = recovered_at - start

    gains = np.column_stack([_array(trace, key) for key in ("kp", "ki", "kd")])
    pre_gains = gains[pre_mask]
    post_gains = gains[post_mask]
    gain_response_time = None
    post_gain_change = None
    chatter = None
    if pre_gains.size and post_gains.size and np.isfinite(pre_gains).all():
        baseline = np.median(pre_gains, axis=0)
        delta = np.linalg.norm(post_gains - baseline, axis=1)
        meaningful = max(0.01, 0.05 * float(np.linalg.norm(baseline)))
        response_indices = np.flatnonzero(delta >= meaningful)
        if response_indices.size:
            post_times = time_s[post_mask]
            gain_response_time = float(post_times[response_indices[0]] - start)
        post_gain_change = float(np.max(delta))
        if len(post_gains) > 1:
            chatter = float(np.mean(np.linalg.norm(np.diff(post_gains, axis=0), axis=1)))

    return {
        "dynamic_event": True,
        "pre_median_error_m": pre_median,
        "pre_peak_error_m": pre_peak,
        "post_peak_error_m": post_peak,
        "post_event_iae": post_iae,
        "post_pre_error_ratio": ratio,
        "recovery_threshold_m": threshold,
        "recovery_time_s": recovery_time,
        "failed_to_recover": recovered_at is None,
        "gain_response_time_s": gain_response_time,
        "post_event_gain_change": post_gain_change,
        "post_event_gain_chatter": chatter,
    }


PAIR_METRICS = (
    "finished",
    "mean_distance_m",
    "max_distance_m",
    "itae",
    "duration_s",
    "saturation_fraction",
    "allocation_limited_fraction",
    "post_peak_error_m",
    "post_event_iae",
    "recovery_time_s",
    "post_event_gain_chatter",
)


def _numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def paired_row(
    scenario: Scenario,
    fixed: dict[str, Any],
    ppo: dict[str, Any],
) -> dict[str, Any]:
    """Return one complete pair. Every delta uses the visible PPO-fixed sign."""
    if scenario.event.kind == "mass_step":
        family = "mass_step"
        severity = abs(scenario.event.value - scenario.base_physics.mass)
        scenario_family = "dynamic"
    elif scenario.event.kind == "force_pulse":
        family = "force_pulse"
        severity = abs(scenario.event.value)
        scenario_family = "dynamic"
    elif "mass" in scenario.tags:
        family = "stationary_mass"
        severity = abs(scenario.base_physics.mass - config.NOMINAL_MASS_KG)
        scenario_family = "stationary"
    elif "friction" in scenario.tags:
        family = "stationary_friction"
        severity = abs(scenario.base_physics.friction_scale - config.NOMINAL_FRICTION)
        scenario_family = "stationary"
    elif "actuator" in scenario.tags:
        family = "stationary_actuator"
        severity = abs(scenario.base_physics.actuator_scale - config.NOMINAL_ACTUATOR)
        scenario_family = "stationary"
    else:
        family = "nominal"
        severity = 0.0
        scenario_family = "nominal"
    row: dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "path_key": scenario.path_key,
        "target_speed": scenario.target_speed,
        "seed": scenario.seed,
        "event_kind": scenario.event.kind,
        "event_value": scenario.event.value,
        "scenario_family": scenario_family,
        "disturbance_family": family,
        "severity_value": severity,
        "base_mass": scenario.base_physics.mass,
        "friction_scale": scenario.base_physics.friction_scale,
        "actuator_scale": scenario.base_physics.actuator_scale,
        "tags": "|".join(scenario.tags),
        "delta_sign": "ppo_minus_fixed",
    }
    for metric in PAIR_METRICS:
        fixed_value = _numeric(fixed.get(metric))
        ppo_value = _numeric(ppo.get(metric))
        row[f"fixed_{metric}"] = fixed.get(metric)
        row[f"ppo_{metric}"] = ppo.get(metric)
        if fixed_value is None or ppo_value is None:
            row[f"delta_{metric}"] = None
            row[f"delta_pct_{metric}"] = None
        else:
            delta = ppo_value - fixed_value
            row[f"delta_{metric}"] = delta
            row[f"delta_pct_{metric}"] = (
                None if abs(fixed_value) < 1e-12 else 100.0 * delta / abs(fixed_value)
            )
    return row


def aggregate_pairs(rows: Iterable[dict[str, Any]], expected_pairs: int) -> dict[str, Any]:
    pairs = list(rows)
    aggregate: dict[str, Any] = {
        "expected_pairs": int(expected_pairs),
        "complete_pairs": len(pairs),
        "missing_pairs": int(expected_pairs) - len(pairs),
        "delta_sign": "ppo_minus_fixed",
        "one_seed_warning": True,
        "practical_improvement_threshold_pct": config.ADAPTATION_MEANINGFUL_IMPROVEMENT_PCT,
    }
    for metric in PAIR_METRICS:
        values = [_numeric(row.get(f"delta_{metric}")) for row in pairs]
        finite = [value for value in values if value is not None]
        aggregate[f"mean_delta_{metric}"] = (
            None if not finite else float(np.mean(finite))
        )
    aggregate["fixed_completion_rate"] = (
        None if not pairs else float(np.mean([bool(row["fixed_finished"]) for row in pairs]))
    )
    aggregate["ppo_completion_rate"] = (
        None if not pairs else float(np.mean([bool(row["ppo_finished"]) for row in pairs]))
    )
    return aggregate
