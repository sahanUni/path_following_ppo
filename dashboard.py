"""Direct Fixed PID versus PPO path comparison dashboard."""

from __future__ import annotations

import argparse
from functools import lru_cache
import math
from pathlib import Path
import subprocess
import sys
from threading import Lock
from typing import Any

from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
import numpy as np
import plotly.graph_objects as go
from stable_baselines3 import PPO

from calibration import GainCalibration
import config
from confirm_advantage import resolve_arms
from core import paths
from core.dynamics import CORRIDOR_M
from env import PathFollowingEnv
from rollout import fixed_policy, run_episode


INK, MUTED, ACCENT = "#e8ecf4", "#93a1b8", "#4fc3f7"
PANEL, LINE = "#1a2233", "#2c3a57"
GOOD, BAD, WARN = "#66d9a3", "#ff7a7a", "#ffce6b"
PATH_COLOR, PPO_COLOR = "#8b97ad", "#c792ea"
# One per model panel, in load order. Distinct enough to tell three
# trajectories apart on the same corridor plot.
MODEL_COLORS = ("#c792ea", "#ffb26b", "#6bd6c4")
MASS_TINT, FORCE_TINT, GUST_TINT = "#ffce6b", "#c792ea", "#8fb8ff"
EVENT_TINTS = {"mass_step": MASS_TINT, "force_pulse": FORCE_TINT, "force_gust": GUST_TINT}
EVENT_LABELS = {
    "mass_step": "mass change",
    "force_pulse": "lateral force",
    "force_gust": "wind gusts",
}
CORRIDOR_GRID = 240
CORRIDOR_STRIDE = 5

PLANT_CONTROLS = (
    ("mass", "Initial mass (kg)", 5.0, 100.0, 0.5, 10.0),
    ("friction", "Floor friction", 0.1, 2.0, 0.05, 1.0),
    ("actuator", "Actuator scale", 0.4, 3.0, 0.05, 1.0),
)
MASS_CONTROLS = (
    ("mass-target", "Target mass (kg)", 5.0, 100.0, 0.5, 25.0),
    ("mass-time", "Change at (s)", 0.0, 60.0, 0.1, 3.0),
)
FORCE_CONTROLS = (
    ("force", "Lateral force (N)", -100.0, 100.0, 1.0, -20.0),
    ("force-start", "Force from (s)", 0.0, 60.0, 0.1, 3.0),
    ("force-end", "Force to (s)", 0.0, 60.0, 0.1, 8.0),
)
GUST_CONTROLS = (
    # Measured on the arc at 0.3 m/s: below 5 N nothing moves, 8-15 N is the
    # band where the corridor is still held but the error is visibly working,
    # and somewhere past 15 N the car is blown out of the corridor outright.
    # The slider stops at 25 so the whole of that is reachable without the
    # interesting part being squeezed into the first fifth of the track.
    ("gust", "Gust strength (N)", 0.0, 25.0, 0.25, 8.0),
    ("gust-tau", "Gust correlation (s)", 0.05, 10.0, 0.05, 1.0),
    ("gust-start", "Gusts from (s)", 0.0, 60.0, 0.1, 0.0),
    ("gust-end", "Gusts to (s)", 0.0, 60.0, 0.1, 30.0),
)
# Dead time in MILLISECONDS and sensor noise in MILLIMETRES, because those are
# the units the numbers are legible in -- 0.0004 m on a slider reads as zero.
# Both are converted back to SI in compare().
#
# The ranges deliberately span past the useful band so the cliff is reachable:
# dead time does nothing until ~60 ms, inverts the Kp ranking by 100 ms, and
# takes most of the gain box out by 200 ms. Sensor noise acts through Kd, and
# at 1 mm the baseline Kd fails outright. Being able to drive the car off the
# path is the point -- that is the tradeoff made visible.
IMPERFECTION_CONTROLS = (
    ("delay", "Actuator dead time (ms)", 0.0, 200.0, 2.0, 1000.0 * config.EVAL_ACTUATOR_DELAY_S),
    ("noise", "Sensor noise (mm)", 0.0, 1.0, 0.025, 1000.0 * config.EVAL_SENSOR_NOISE_M),
)
ALL_CONTROLS = (
    PLANT_CONTROLS + MASS_CONTROLS + FORCE_CONTROLS + GUST_CONTROLS + IMPERFECTION_CONTROLS
)

# Slider bounds for the hand-tuned Fixed PID. Deliberately the CALIBRATION
# SEARCH space, not the narrower PPO action box: the point of the left panel is
# to let you try gains the policy cannot reach and see for yourself where the
# box should have been.
GAIN_STEPS = (0.25, 0.05, 0.05)
GAIN_LABELS = ("Kp (steering)", "Ki (steering)", "Kd (steering)")


def gain_controls(calibration: GainCalibration) -> tuple[tuple[Any, ...], ...]:
    """Gain sliders, defaulted to the calibrated baseline for this artifact."""
    return tuple(
        (
            f"gain-{name}",
            GAIN_LABELS[index],
            float(config.CALIBRATION_SEARCH_LOW[index]),
            float(config.CALIBRATION_SEARCH_HIGH[index]),
            GAIN_STEPS[index],
            float(calibration.base[index]),
        )
        for index, name in enumerate(("kp", "ki", "kd"))
    )


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return parsed


def resolve_calibration(
    project_dir: str | Path, calibration_path: str | Path | None = None
) -> Path:
    """Newest available calibration artifact, or the one explicitly named."""
    project = Path(project_dir)
    if calibration_path is not None:
        return (project / calibration_path).resolve(strict=True)
    for candidate in DirectComparisonRunner.CALIBRATION_CANDIDATES:
        resolved = project / candidate
        if resolved.exists():
            return resolved.resolve(strict=True)
    raise FileNotFoundError(
        "No calibration artifact found. Run calibrate_pid.py first; tried "
        + ", ".join(str(c) for c in DirectComparisonRunner.CALIBRATION_CANDIDATES)
    )


class DirectComparisonRunner:
    """Load the two frozen controllers once and run matched selected episodes."""

    # A model is only meaningful alongside the calibration it was trained
    # against: the action vector is interpreted through low/base/high, so
    # pairing a model with a different gain box silently changes what its
    # outputs mean. These two move together or not at all.
    # Newest first, and v6 is deliberately NOT in this list. Its sweep selected
    # a gain that finishes every scenario by crawling (Kp 1.23, 5.25 cm mean
    # error against 1.88 cm for the best tracker) and its box tops out at
    # Kp 2.25, far below where the plant is controllable. Loading it by default
    # would quietly show a broken baseline.
    # (label, path) tried in order. The two arms are shown side by side because
    # the whole point of the comparison is that they behave differently despite
    # one of them being handed the plant parameters outright.
    PREFERRED_MODELS = (
        ("PPO (blind)", Path("runs") / "rq1_blind" / "seed21" / "best_model.zip"),
        ("PPO (context)", Path("runs") / "rq2_context" / "seed42" / "best_model.zip"),
    )
    # Used only when NONE of the preferred ones are present. Otherwise a stale
    # single-arm model would sit alongside the two arms as a third panel and
    # muddle the comparison the dashboard exists to show.
    FALLBACK_MODELS = (
        ("PPO", Path("runs") / "v7_seed7" / "final_model.zip"),
        ("PPO", Path("runs") / "development" / "best_model.zip"),
    )
    MODEL_CANDIDATES = PREFERRED_MODELS + FALLBACK_MODELS
    DEFAULT_MODEL = PREFERRED_MODELS[0][1]
    # The dashboard is useful before a model exists -- the whole left panel is
    # hand-tuning against the live plant -- so a missing model is a degraded
    # mode, not an error. A missing calibration is fatal: there would be no
    # baseline gains to default the sliders to.
    CALIBRATION_CANDIDATES = (
        Path("runs") / "calibration_v7.json",
        Path("runs") / "calibration_v5.json",
        Path("runs") / "calibration_twosided.json",
    )

    def __init__(
        self,
        project_dir: str | Path,
        model_path: str | Path | None = None,
        calibration_path: str | Path | None = None,
    ) -> None:
        self.project_dir = Path(project_dir).resolve(strict=True)
        self.calibration_path = self._resolve_calibration(calibration_path)
        if self.calibration_path.suffix.lower() != ".json":
            raise ValueError("the configured PID calibration is not a JSON artifact")
        self.calibration = GainCalibration.load(self.calibration_path)

        self.models: list[dict[str, Any]] = []
        self.model_error = ""
        for label, path in self._wanted_models(model_path):
            resolved = self.project_dir / path
            if not resolved.exists():
                if model_path is not None:
                    # Explicitly asked for by name: a typo must not silently
                    # degrade into a missing panel.
                    raise FileNotFoundError(f"No PPO model at {resolved}")
                continue
            if resolved.suffix.lower() != ".zip":
                raise ValueError(f"not a ZIP artifact: {resolved}")
            # Each model carries its own observation arms. A blind model and a
            # plant-context model need environments of different widths, so the
            # arms travel with the model rather than being a global setting.
            arms = resolve_arms(resolved.resolve())
            model = PPO.load(resolved, device="cpu")
            # A missing arms.json defaults to blind, which is right for runs
            # made before train.py started writing one -- but silently wrong
            # for a context model whose manifest went astray. Checked here so
            # it reports the artifact and the fix, rather than failing later
            # inside SB3 with a bare shape error from a callback.
            probe = PathFollowingEnv(
                calibration=self.calibration, training=False,
                preview=arms["preview"], plant_context=arms["plant_context"],
            )
            width = probe.observation_space.shape[0]
            probe.close()
            expected = model.observation_space.shape[0]
            if width != expected:
                raise ValueError(
                    f"{resolved.name} expects an observation of {expected} values "
                    f"but arms {arms} build one of {width}. The arms.json beside "
                    f"the model is missing or wrong: {resolved.parent / 'arms.json'}"
                )
            self.models.append({
                "label": label,
                "path": resolved.resolve(strict=True),
                "arms": arms,
                "model": model,
            })
        if not self.models:
            self.model_error = (
                "No PPO model found. Train one, then reload:  python train.py "
                f"--calibration {self.calibration_path.name} --run-dir runs/development"
            )

        # Separate caches, because the panels have different dependencies: the
        # Fixed PID re-runs whenever a gain slider moves, and the model episodes
        # do not depend on those gains at all. One shared key would re-run every
        # slow panel on each keystroke.
        self._fixed_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._model_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._lock = Lock()

    @property
    def model_path(self) -> Path | None:
        """First loaded model, for callers that only need one (e.g. replay)."""
        return self.models[0]["path"] if self.models else None

    @property
    def model(self):
        return self.models[0]["model"] if self.models else None

    def _wanted_models(
        self, model_path: str | Path | None
    ) -> list[tuple[str, Path]]:
        if model_path is not None:
            requested = [model_path] if isinstance(model_path, (str, Path)) else list(model_path)
            out: list[tuple[str, Path]] = []
            for item in requested:
                text = str(item)
                # "label=path" lets the caller name a panel; bare paths are
                # named from the run directory, which is usually meaningful
                # here (rq1_blind/seed21 -> "rq1_blind seed21").
                if "=" in text and not Path(text).exists():
                    label, _, raw = text.partition("=")
                    out.append((label.strip(), Path(raw.strip())))
                else:
                    path = Path(text)
                    out.append((f"{path.parent.parent.name} {path.parent.name}".strip(), path))
            return out
        preferred = [
            (label, path) for label, path in self.PREFERRED_MODELS
            if (self.project_dir / path).exists()
        ]
        return preferred or list(self.FALLBACK_MODELS)

    def _resolve_calibration(self, calibration_path: str | Path | None) -> Path:
        return resolve_calibration(self.project_dir, calibration_path)

    def compare(
        self,
        path_key: str,
        target_speed: Any,
        mass: Any = 10.0,
        friction: Any = 1.0,
        actuator: Any = 1.0,
        mass_enabled: bool = False,
        mass_target: Any = 25.0,
        mass_time: Any = 3.0,
        force_enabled: bool = False,
        force: Any = -20.0,
        force_start: Any = 3.0,
        force_end: Any = 8.0,
        gust_enabled: bool = False,
        gust: Any = 12.0,
        gust_tau: Any = 1.0,
        gust_start: Any = 0.0,
        gust_end: Any = 30.0,
        delay_ms: Any = None,
        noise_mm: Any = None,
        kp: Any = None,
        ki: Any = None,
        kd: Any = None,
    ) -> dict[str, Any]:
        if path_key not in paths.CATALOGUE:
            raise ValueError("select a valid path")
        speed = _number(target_speed, "target speed", 0.3, 0.7)
        if speed not in config.SPEED_TARGETS:
            raise ValueError("select a valid target speed")
        initial_mass = _number(mass, "initial mass", 5.0, 100.0)
        floor_friction = _number(friction, "floor friction", 0.1, 2.0)
        actuator_scale = _number(actuator, "actuator scale", 0.4, 3.0)
        target_mass = _number(mass_target, "target mass", 5.0, 100.0)
        change_time = _number(mass_time, "mass-change time", 0.0, 60.0)
        force_amount = _number(force, "lateral force", -100.0, 100.0)
        force_from = _number(force_start, "force start", 0.0, 60.0)
        force_to = _number(force_end, "force end", 0.0, 60.0)
        gust_sigma = _number(gust, "gust strength", 0.0, 25.0)
        gust_correlation = _number(gust_tau, "gust correlation", 0.05, 10.0)
        gust_from = _number(gust_start, "gust start", 0.0, 60.0)
        gust_to = _number(gust_end, "gust end", 0.0, 60.0)
        delay_s = 0.001 * _number(
            0.0 if delay_ms is None else delay_ms, "dead time", 0.0, 200.0
        )
        noise_m = 0.001 * _number(
            0.0 if noise_mm is None else noise_mm, "sensor noise", 0.0, 1.0
        )
        gains = np.array(
            [
                _number(
                    self.calibration.base[index] if raw is None else raw,
                    name,
                    float(config.CALIBRATION_SEARCH_LOW[index]),
                    float(config.CALIBRATION_SEARCH_HIGH[index]),
                )
                for index, (raw, name) in enumerate(
                    ((kp, "Kp"), (ki, "Ki"), (kd, "Kd"))
                )
            ],
            dtype=np.float64,
        )

        ideal_time = float(paths.get(path_key)["length"]) / speed
        episode_limit = config.TIME_LIMIT_FACTOR * ideal_time + config.TIME_LIMIT_MARGIN_S
        if mass_enabled and change_time >= episode_limit:
            raise ValueError(f"mass change must occur before {episode_limit:.1f} s")
        if force_enabled:
            if force_to <= force_from:
                raise ValueError("force end must be later than force start")
            if force_from >= episode_limit:
                raise ValueError(f"force must start before {episode_limit:.1f} s")
        if gust_enabled:
            if gust_to <= gust_from:
                raise ValueError("gusts must end later than they start")
            if gust_from >= episode_limit:
                raise ValueError(f"gusts must start before {episode_limit:.1f} s")

        events: list[dict[str, Any]] = []
        if mass_enabled:
            events.append(
                {
                    "kind": "mass_step",
                    "start_s": change_time,
                    "end_s": None,
                    "amount": target_mass,
                }
            )
        if force_enabled:
            events.append(
                {
                    "kind": "force_pulse",
                    "start_s": force_from,
                    "end_s": force_to,
                    "amount": force_amount,
                }
            )
        if gust_enabled:
            events.append(
                {
                    "kind": "force_gust",
                    "start_s": gust_from,
                    "end_s": gust_to,
                    "amount": gust_sigma,
                    "tau_s": gust_correlation,
                    # Fixed seed: both controllers must meet the same wind, and
                    # the same sliders must reproduce the same run tomorrow.
                    "seed": config.SEED,
                }
            )
        if not events:
            events.append(
                {"kind": "none", "start_s": 0.0, "end_s": None, "amount": 0.0}
            )

        # The scenario is everything both controllers share. The gains are NOT
        # part of it, which is exactly why the PPO side can be cached against
        # this key alone and skipped while you drag a gain slider.
        scenario_key = (
            path_key,
            speed,
            initial_mass,
            floor_friction,
            actuator_scale,
            bool(mass_enabled),
            target_mass,
            change_time,
            bool(force_enabled),
            force_amount,
            force_from,
            force_to,
            bool(gust_enabled),
            gust_sigma,
            gust_correlation,
            gust_from,
            gust_to,
            delay_s,
            noise_m,
        )
        options = {
            "path_key": path_key,
            "v_target": speed,
            "mass": initial_mass,
            "friction": floor_friction,
            "actuator": actuator_scale,
            "actuator_delay_s": delay_s,
            "sensor_noise_m": noise_m,
            # Fixed, like the gust seed: both controllers must meet the same
            # sensor trace, and the same sliders must reproduce the same run
            # tomorrow.
            "noise_seed": config.SEED,
            "stage": 3,
            "disturbances": events,
        }

        with self._lock:
            fixed_key = scenario_key + tuple(gains.tolist())
            fixed = self._fixed_cache.get(fixed_key)
            if fixed is None:
                fixed = self._run_episode(path_key, options, gains=gains)
                self._fixed_cache[fixed_key] = fixed

            models = []
            for entry in self.models:
                key = scenario_key + (str(entry["path"]),)
                result = self._model_cache.get(key)
                if result is None:
                    result = self._run_episode(path_key, options, entry=entry)
                    self._model_cache[key] = result
                models.append({**result, "label": entry["label"], "arms": entry["arms"]})

        return {
            "path_key": path_key,
            "target_speed": speed,
            "events": events,
            "gains": gains,
            "delay_s": delay_s,
            "noise_m": noise_m,
            "fixed": fixed,
            "models": models,
        }

    def _run_episode(
        self,
        path_key: str,
        options: dict[str, Any],
        *,
        gains: np.ndarray | None = None,
        entry: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """One episode: fixed gains, or the policy in `entry`."""
        arms = entry["arms"] if entry else {"preview": False, "plant_context": False}
        env = PathFollowingEnv(
            calibration=self.calibration,
            training=False,
            path_keys=(path_key,),
            fixed_gains=gains,
            preview=arms["preview"],
            plant_context=arms["plant_context"],
        )

        def ppo_policy(observation: np.ndarray) -> np.ndarray:
            action, _ = entry["model"].predict(observation, deterministic=True)
            return np.asarray(action, dtype=np.float32)

        policy = fixed_policy if entry is None else ppo_policy
        try:
            metrics, trace = run_episode(
                env, policy, seed=config.SEED, options=options
            )
        finally:
            env.close()
        return {"metrics": metrics, "trace": trace}


def _base_figure(title: str, y_label: str, *, height: int, x_label: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=55, r=15, t=42, b=40),
        paper_bgcolor=PANEL,
        plot_bgcolor="#0a0f1a",
        title=dict(text=title, x=0.01, font=dict(size=13, color=MUTED)),
        showlegend=False,
        hovermode="x unified",
    )
    figure.update_xaxes(title=x_label, gridcolor="#1e2a42", zeroline=False)
    figure.update_yaxes(title=y_label, gridcolor="#1e2a42", zeroline=False)
    return figure


def _corridor_field(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    marks = points[::CORRIDOR_STRIDE]
    padding = CORRIDOR_M * 1.35
    low, high = points.min(axis=0) - padding, points.max(axis=0) + padding
    grid_x = np.linspace(low[0], high[0], CORRIDOR_GRID)
    grid_y = np.linspace(low[1], high[1], CORRIDOR_GRID)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    distance_squared = np.full(mesh_x.shape, np.inf)
    for point_x, point_y in marks:
        np.minimum(
            distance_squared,
            (mesh_x - point_x) ** 2 + (mesh_y - point_y) ** 2,
            out=distance_squared,
        )
    return grid_x, grid_y, np.sqrt(distance_squared)


@lru_cache(maxsize=len(paths.CATALOGUE))
def _corridor(path_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _corridor_field(paths.get(path_key)["pts"])


def _values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def _wind_arrows(figure: go.Figure, rows: list[dict[str, Any]], count: int = 22) -> None:
    """Draw the wind as arrows along the trajectory, tail on the car.

    Magnitude alone (the disturbance chart) cannot show that a gust which veers
    across the path costs far more than one blowing along it. Direction is the
    half of the disturbance the tracking error actually responds to, so it
    belongs on the plot where the deviation is visible.
    """
    forces = [
        (row.get("x"), row.get("y"), row.get("external_force_x"), row.get("external_force_y"))
        for row in rows
    ]
    forces = [
        f for f in forces
        if None not in f and float(np.hypot(f[2], f[3])) > 1e-9
    ]
    if not forces:
        return
    stride = max(1, len(forces) // count)
    sampled = forces[::stride]
    peak = max(float(np.hypot(f[2], f[3])) for f in sampled)
    if peak <= 0:
        return
    # Scaled so the strongest arrow is a fixed fraction of the path extent,
    # which keeps arrows readable whether the gust is 2 N or 25 N.
    extent = float(np.max(np.ptp(np.array([[f[0], f[1]] for f in forces]), axis=0)))
    scale = max(extent, 1.0) * 0.10 / peak
    for x, y, fx, fy in sampled:
        figure.add_annotation(
            x=x + fx * scale, y=y + fy * scale, ax=x, ay=y,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.2,
            arrowcolor=GUST_TINT, opacity=0.75, text="",
        )


def velocity_figure(
    rows: list[dict[str, Any]], events: list[dict[str, Any]] | None = None
) -> go.Figure:
    """Achieved speed against target, with the authority conflict shaded.

    Steering has first claim on the wheels, so the speed channel is what gives
    way in a corner -- the car brakes itself into apexes without anything asking
    it to. Shading the allocation-limited samples makes that visible instead of
    leaving it as an unexplained dip.
    """
    figure = _base_figure(
        "speed - achieved vs target", "v (m/s)", height=250, x_label="time (s)"
    )
    times = _values(rows, "t")
    trace_end = times[-1] if times else 0.0
    for event in events or []:
        if event["kind"] == "none" or float(event["start_s"]) > trace_end:
            continue
        end = event.get("end_s")
        figure.add_vrect(
            x0=float(event["start_s"]),
            x1=min(float(end) if end is not None else trace_end, trace_end),
            fillcolor=EVENT_TINTS.get(event["kind"], FORCE_TINT),
            opacity=0.10, line_width=0,
        )
    targets = _values(rows, "v_target")
    if targets:
        figure.add_hline(
            y=targets[0], line=dict(color=MUTED, width=1, dash="dot"),
            annotation_text="target", annotation_position="top left",
            annotation_font=dict(size=10, color=MUTED),
        )
    # Contiguous runs, not one shape per sample. An episode is ~1000 control
    # steps and most of them can be allocation-limited, which is a thousand
    # plotly shapes and a chart that takes seconds to draw. Merging leaves a
    # handful of spans that say the same thing.
    if times:
        step = (times[-1] - times[0]) / max(len(times) - 1, 1)
        spans: list[list[float]] = []
        for row in rows:
            moment = row.get("t")
            if moment is None:
                continue
            if not bool(row.get("allocation_limited")):
                continue
            if spans and moment - spans[-1][1] <= step * 1.5:
                spans[-1][1] = moment + step
            else:
                spans.append([moment, moment + step])
        for start, end in spans[:200]:
            figure.add_vrect(
                x0=start, x1=end,
                fillcolor=WARN, opacity=0.16, line_width=0, layer="below",
            )
        if spans:
            figure.add_annotation(
                text=f"shaded: steering has taken wheel authority ({len(spans)} spans)",
                showarrow=False, font=dict(size=10, color=WARN),
                xref="paper", yref="paper", x=0.99, y=1.13, xanchor="right",
            )
    figure.add_scatter(
        x=times, y=_values(rows, "speed"), mode="lines",
        line=dict(color=GOOD, width=1.8), name="speed",
    )
    figure.update_layout(
        showlegend=True,
        legend=dict(orientation="h", y=1.18, x=0.18, font=dict(size=10)),
    )
    return figure


def disturbance_figure(rows: list[dict[str, Any]]) -> go.Figure:
    """External force on the body: total magnitude plus its two components.

    The components matter because the sign flips are what the steering PID has
    to chase; a gust of constant magnitude that keeps veering is a much harder
    problem than a steady push of the same size.
    """
    figure = _base_figure(
        "external force - wind and applied disturbances",
        "F (N)", height=250, x_label="time (s)",
    )
    times = _values(rows, "t")
    force_x = _values(rows, "external_force_x")
    force_y = _values(rows, "external_force_y")
    if not times or not force_x or not force_y:
        figure.add_annotation(
            text="no external disturbance in this episode",
            showarrow=False, font=dict(size=12, color=MUTED),
            xref="paper", yref="paper", x=0.5, y=0.5,
        )
        return figure
    magnitude = list(np.hypot(np.array(force_x), np.array(force_y)))
    figure.add_hline(y=0.0, line=dict(color=LINE, width=1))
    figure.add_scatter(
        x=times, y=magnitude, mode="lines",
        line=dict(color=GUST_TINT, width=2.0), name="|F|",
    )
    figure.add_scatter(
        x=times, y=force_x, mode="lines",
        line=dict(color=MUTED, width=1.0, dash="dot"), name="Fx (world)",
    )
    figure.add_scatter(
        x=times, y=force_y, mode="lines",
        line=dict(color=PPO_COLOR, width=1.0, dash="dot"), name="Fy (world)",
    )
    figure.update_layout(
        showlegend=True,
        legend=dict(orientation="h", y=1.18, x=0.10, font=dict(size=10)),
    )
    return figure


def gain_figure(rows: list[dict[str, Any]], calibration: GainCalibration) -> go.Figure:
    """What the controller did with its gains, against the box it may use.

    Flat lines here are the null result this project keeps rediscovering, so it
    is worth being able to see it directly rather than inferring it from a mean.
    """
    figure = _base_figure(
        "steering gains - applied over the episode", "gain", height=250, x_label="time (s)"
    )
    times = _values(rows, "t")
    for bound, dash in ((calibration.low[0], "dot"), (calibration.high[0], "dot")):
        figure.add_hline(y=float(bound), line=dict(color=LINE, width=1, dash=dash))
    for key, name, color in (
        ("kp", "Kp", ACCENT), ("ki", "Ki", GOOD), ("kd", "Kd", WARN)
    ):
        figure.add_scatter(
            x=times, y=_values(rows, key), mode="lines",
            line=dict(color=color, width=1.6), name=name,
        )
    figure.update_layout(
        showlegend=True,
        legend=dict(orientation="h", y=1.18, x=0.18, font=dict(size=10)),
    )
    return figure


def trajectory_figure(
    rows: list[dict[str, Any]],
    path_key: str,
    controller_name: str,
    color: str,
    show_wind: bool = True,
) -> go.Figure:
    reference = paths.get(path_key)
    points = reference["pts"]
    figure = _base_figure(
        f"trajectory - {reference['name']}", "y (m)", height=430, x_label="x (m)"
    )
    grid_x, grid_y, distance = _corridor(path_key)
    figure.add_contour(
        x=grid_x,
        y=grid_y,
        z=distance,
        showscale=False,
        hoverinfo="skip",
        name="corridor",
        line=dict(width=0),
        contours=dict(type="constraint", operation="<", value=CORRIDOR_M),
        fillcolor="rgba(139,151,173,0.10)",
    )
    figure.add_scatter(
        x=points[::10, 0],
        y=points[::10, 1],
        mode="lines",
        line=dict(color=PATH_COLOR, width=1.6, dash="dash"),
        hoverinfo="skip",
        name="reference path",
    )
    figure.add_scatter(
        x=_values(rows, "x"),
        y=_values(rows, "y"),
        mode="lines",
        line=dict(color=color, width=2.4),
        name=controller_name,
    )
    figure.add_scatter(
        x=[points[0, 0]],
        y=[points[0, 1]],
        mode="markers",
        marker=dict(color=GOOD, size=10),
        name="start",
    )
    figure.add_scatter(
        x=[points[-1, 0]],
        y=[points[-1, 1]],
        mode="markers",
        marker=dict(color=WARN, size=10, symbol="square"),
        name="finish",
    )
    if show_wind:
        _wind_arrows(figure, rows)
    figure.update_layout(hovermode="closest")
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


def deviation_figure(
    rows: list[dict[str, Any]], events: list[dict[str, Any]] | None = None
) -> go.Figure:
    figure = _base_figure(
        "tracking error - e_ct vs true distance off path",
        "e (m)",
        height=285,
        x_label="time (s)",
    )
    trace_end = _values(rows, "t")[-1] if rows else 0.0
    for event in events or []:
        if event["kind"] == "none" or float(event["start_s"]) > trace_end:
            continue
        end = event.get("end_s")
        tint = EVENT_TINTS.get(event["kind"], FORCE_TINT)
        label = EVENT_LABELS.get(event["kind"], event["kind"])
        figure.add_vrect(
            x0=float(event["start_s"]),
            x1=min(float(end) if end is not None else trace_end, trace_end),
            fillcolor=tint,
            opacity=0.10,
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
            annotation_font=dict(size=10, color=tint),
        )
    for limit in (CORRIDOR_M, -CORRIDOR_M):
        figure.add_hline(y=limit, line=dict(color=BAD, width=1, dash="dot"))
    figure.add_hline(y=0.0, line=dict(color=LINE, width=1))
    times = _values(rows, "t")
    figure.add_scatter(
        x=times,
        y=_values(rows, "dist"),
        mode="lines",
        line=dict(color=PATH_COLOR, width=1.2, dash="dot"),
        name="distance off path",
    )
    figure.add_scatter(
        x=times,
        y=_values(rows, "e_ct"),
        mode="lines",
        line=dict(color=ACCENT, width=1.8),
        name="e_ct (controller)",
    )
    figure.update_layout(
        showlegend=True,
        legend=dict(orientation="h", y=1.16, x=0.18, font=dict(size=10)),
    )
    return figure


def _metric(label: str, value: str) -> html.Div:
    return html.Div([html.B(label), html.Span(value)], className="metric")


def _distance(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f} cm"
    except (TypeError, ValueError):
        return "n/a"


def _duration(value: Any) -> str:
    try:
        return f"{float(value):.2f} s"
    except (TypeError, ValueError):
        return "n/a"


def controller_metrics(metrics: dict[str, Any]) -> html.Div:
    return html.Div(
        [
            _metric("Status", "Finished" if metrics.get("finished") else "Not finished"),
            _metric("Mean deviation", _distance(metrics.get("mean_distance_m"))),
            _metric("Maximum deviation", _distance(metrics.get("max_distance_m"))),
            _metric("Duration", _duration(metrics.get("duration_s"))),
        ],
        className="metrics",
    )


def _gain_activity(rows: list[dict[str, Any]]) -> str:
    """How much the policy actually MOVED each gain, not just its average.

    A scheduler that settles on one constant is the null result this project
    keeps rediscovering, and a mean alone hides it perfectly. The spread is
    what tells the two apart at a glance.
    """
    parts = []
    for key, name in (("kp", "Kp"), ("ki", "Ki"), ("kd", "Kd")):
        values = _values(rows, key)
        if not values:
            continue
        low, high = min(values), max(values)
        span = high - low
        parts.append(
            f"{name} {np.mean(values):.2f}"
            + (f" ({low:.2f}-{high:.2f})" if span > 1e-6 else " (constant)")
        )
    return "  ·  ".join(parts) if parts else "no gains recorded"


def _panel(
    title: str,
    note: str,
    result: dict[str, Any],
    path_key: str,
    events: list[dict[str, Any]],
    color: str,
    calibration: GainCalibration,
) -> html.Section:
    rows = result["trace"]
    graph = {"displaylogo": False, "responsive": True}
    return html.Section(
        [
            html.H2(title),
            html.P(note, className="controller-note"),
            controller_metrics(result["metrics"]),
            dcc.Graph(figure=trajectory_figure(rows, path_key, title, color), config=graph),
            dcc.Graph(figure=deviation_figure(rows, events), config=graph),
            dcc.Graph(figure=velocity_figure(rows, events), config=graph),
            dcc.Graph(figure=disturbance_figure(rows), config=graph),
            dcc.Graph(figure=gain_figure(rows, calibration), config=graph),
        ],
        className="comparison-panel",
    )


def comparison_view(result: dict[str, Any], runner: "DirectComparisonRunner") -> html.Div:
    path_key = result["path_key"]
    events = result["events"]
    kp, ki, kd = result["gains"]
    baseline = runner.calibration.base
    drift = float(np.max(np.abs(np.asarray(result["gains"]) - baseline)))
    fixed_note = f"Kp {kp:.3f}, Ki {ki:.3f}, Kd {kd:.3f}"
    if drift > 1e-9:
        fixed_note += (
            f"  ·  hand-tuned (baseline Kp {baseline[0]:.3f}, "
            f"Ki {baseline[1]:.3f}, Kd {baseline[2]:.3f})"
        )
    else:
        fixed_note += f"  ·  calibrated baseline from {runner.calibration_path.name}"

    panels = [
        _panel(
            "Fixed PID", fixed_note, result["fixed"], path_key, events, ACCENT,
            runner.calibration,
        )
    ]
    if not result["models"]:
        panels.append(
            html.Section(
                [html.H2("PPO"), html.Div(runner.model_error, className="message error")],
                className="comparison-panel",
            )
        )
    for index, entry in enumerate(result["models"]):
        arms = entry["arms"]
        sees = [
            name for name, on in
            (("path preview", arms["preview"]), ("plant context", arms["plant_context"]))
            if on
        ]
        note = _gain_activity(entry["trace"])
        note += "  ·  sees: " + (", ".join(sees) if sees else "error history only")
        panels.append(
            _panel(
                entry["label"], note, entry, path_key, events,
                MODEL_COLORS[index % len(MODEL_COLORS)], runner.calibration,
            )
        )
    return html.Div(panels, className="comparison-grid")


def slider_row(
    control_id: str,
    label: str,
    minimum: float,
    maximum: float,
    step: float,
    default: float,
) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span(label),
                    dcc.Input(
                        id=f"n-{control_id}",
                        type="number",
                        value=default,
                        min=minimum,
                        max=maximum,
                        step=step,
                        debounce=True,
                    ),
                ],
                className="slider-label",
            ),
            dcc.Slider(
                minimum,
                maximum,
                step,
                value=default,
                id=f"s-{control_id}",
                marks=None,
                allow_direct_input=False,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
        className="knob",
    )


def create_app(
    project_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
) -> Dash:
    project = Path(project_dir or Path(__file__).resolve().parent)
    runner = DirectComparisonRunner(project, model_path, calibration_path)
    gains = gain_controls(runner.calibration)
    sync_controls = ALL_CONTROLS + gains
    app = Dash(__name__, title="Fixed PID vs PPO")
    app.layout = html.Div(
        [
            html.Button(
                "☰  Controls", id="panel-toggle", className="panel-toggle", n_clicks=0
            ),
            html.Header(
                [
                    html.H1("Fixed PID vs PPO"),
                    html.P(
                        "The same selected path, plant, and mid-episode manipulation, shown end to end with the calibrated Fixed PID and the best PPO model.",
                        className="kicker",
                    ),
                ],
                className="masthead",
            ),
            html.Aside(
                [
                    html.Div(
                        [
                            html.Div("Controls", className="drawer-title"),
                            html.Button(
                                "✕",
                                id="panel-close",
                                className="drawer-close",
                                n_clicks=0,
                            ),
                        ],
                        className="drawer-head",
                    ),
                    html.Div("Path and speed", className="section-title"),
                    html.Div(
                        [
                            html.Label("Path", htmlFor="path-select"),
                            dcc.Dropdown(
                                id="path-select",
                                value=paths.DEFAULT,
                                options=[
                                    {"label": name, "value": key}
                                    for key, name in paths.options()
                                ],
                                clearable=False,
                            ),
                        ],
                        className="selector",
                    ),
                    html.Div(
                        [
                            html.Label("Target speed", htmlFor="speed-select"),
                            dcc.Dropdown(
                                id="speed-select",
                                value=0.5,
                                options=[
                                    {"label": f"{speed:.1f} m/s", "value": speed}
                                    for speed in config.SPEED_TARGETS
                                ],
                                clearable=False,
                            ),
                        ],
                        className="selector",
                    ),
                    html.Div("Fixed PID gains (left panel)", className="section-title"),
                    html.Div(
                        "Defaults are the calibrated baseline. Move them and the "
                        "left panel re-simulates; the PPO run is untouched, so "
                        "you are always comparing against the same episode.",
                        className="group-hint",
                    ),
                    *[slider_row(*control) for control in gains],
                    html.Button(
                        "Reset to calibrated baseline",
                        id="gain-reset",
                        className="drawer-close",
                        n_clicks=0,
                    ),
                    html.Div("Plant imperfections", className="section-title"),
                    html.Div(
                        "Dead time queues the wheel command; sensor noise "
                        "corrupts only the e_ct the PID reads, never the score. "
                        "With both at zero, raising Kp is free and the two "
                        "controllers converge on the same answer.",
                        className="group-hint",
                    ),
                    *[slider_row(*control) for control in IMPERFECTION_CONTROLS],
                    html.Div("Watch in MuJoCo", className="section-title"),
                    html.Div(
                        "Opens a separate viewer window replaying the current "
                        "settings, with the wind drawn as an arrow over the car. "
                        "It runs as its own process because MuJoCo's viewer "
                        "needs the main thread.",
                        className="group-hint",
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Replay Fixed PID", id="replay-fixed",
                                className="drawer-close", n_clicks=0,
                            ),
                            *[
                                html.Button(
                                    f"Replay {entry['label']}",
                                    id={"role": "replay-model", "index": index},
                                    className="drawer-close", n_clicks=0,
                                )
                                for index, entry in enumerate(runner.models)
                            ],
                        ],
                        className="selector",
                    ),
                    html.Div(id="replay-status", className="group-hint"),
                    html.Div("Initial plant", className="section-title"),
                    *[slider_row(*control) for control in PLANT_CONTROLS],
                    html.Div(
                        [
                            html.Div(
                                dcc.Checklist(
                                    id="mass-enabled",
                                    value=[],
                                    options=[
                                        {"label": " Mass change", "value": "on"}
                                    ],
                                ),
                                className="group-title",
                            ),
                            html.Div(
                                "Persistent absolute target mass",
                                className="group-hint",
                            ),
                            *[slider_row(*control) for control in MASS_CONTROLS],
                        ],
                        id="mass-group",
                        className="control-group off",
                    ),
                    html.Div(
                        [
                            html.Div(
                                dcc.Checklist(
                                    id="force-enabled",
                                    value=[],
                                    options=[
                                        {"label": " External force", "value": "on"}
                                    ],
                                ),
                                className="group-title",
                            ),
                            html.Div(
                                "Vehicle-relative lateral pulse",
                                className="group-hint",
                            ),
                            *[slider_row(*control) for control in FORCE_CONTROLS],
                        ],
                        id="force-group",
                        className="control-group off",
                    ),
                    html.Div(
                        [
                            html.Div(
                                dcc.Checklist(
                                    id="gust-enabled",
                                    value=[],
                                    options=[{"label": " Wind gusts", "value": "on"}],
                                ),
                                className="group-title",
                            ),
                            html.Div(
                                "Seeded world-frame gusts: strength is the per-axis "
                                "spread, correlation is how long a gust holds its "
                                "bearing before it veers.",
                                className="group-hint",
                            ),
                            *[slider_row(*control) for control in GUST_CONTROLS],
                        ],
                        id="gust-group",
                        className="control-group off",
                    ),
                ],
                id="controls-drawer",
                className="controls panel drawer open",
            ),
            html.Main(html.Div(id="comparison-content"), className="results"),
        ],
        className="app",
    )

    @app.callback(
        Output("controls-drawer", "className"),
        Input("panel-toggle", "n_clicks"),
        Input("panel-close", "n_clicks"),
        State("controls-drawer", "className"),
        prevent_initial_call=True,
    )
    def toggle_drawer(_open_clicks: int, _close_clicks: int, current: str):
        closed = "controls panel drawer"
        if ctx.triggered_id == "panel-close":
            return closed
        return closed if "open" in (current or "") else f"{closed} open"

    for control_id, _label, _minimum, _maximum, _step, _default in ALL_CONTROLS:
        app.callback(
            Output(f"s-{control_id}", "value"),
            Output(f"n-{control_id}", "value"),
            Input(f"s-{control_id}", "value"),
            Input(f"n-{control_id}", "value"),
            prevent_initial_call=True,
        )(
            lambda slider_value, number_value, cid=control_id: (
                (number_value, no_update)
                if ctx.triggered_id == f"n-{cid}"
                else (no_update, slider_value)
            )
        )

    # The gain rows carry a third input, the reset button. It has to live in
    # the SAME callback as the slider/number sync: two callbacks writing one
    # Output is a duplicate-output error in Dash.
    for control_id, _label, _minimum, _maximum, _step, default in gains:
        app.callback(
            Output(f"s-{control_id}", "value"),
            Output(f"n-{control_id}", "value"),
            Input(f"s-{control_id}", "value"),
            Input(f"n-{control_id}", "value"),
            Input("gain-reset", "n_clicks"),
            prevent_initial_call=True,
        )(
            lambda slider_value, number_value, _clicks, cid=control_id, base=default: (
                (base, base)
                if ctx.triggered_id == "gain-reset"
                else (number_value, no_update)
                if ctx.triggered_id == f"n-{cid}"
                else (no_update, slider_value)
            )
        )

    @app.callback(
        Output("mass-group", "className"),
        Output("force-group", "className"),
        Output("gust-group", "className"),
        Input("mass-enabled", "value"),
        Input("force-enabled", "value"),
        Input("gust-enabled", "value"),
    )
    def style_event_groups(
        mass_values: list[str], force_values: list[str], gust_values: list[str]
    ):
        def group_class(values: list[str]) -> str:
            return "control-group" if "on" in (values or []) else "control-group off"

        return group_class(mass_values), group_class(force_values), group_class(gust_values)

    @app.callback(
        Output("comparison-content", "children"),
        Input("path-select", "value"),
        Input("speed-select", "value"),
        Input("s-mass", "value"),
        Input("s-friction", "value"),
        Input("s-actuator", "value"),
        Input("mass-enabled", "value"),
        Input("s-mass-target", "value"),
        Input("s-mass-time", "value"),
        Input("force-enabled", "value"),
        Input("s-force", "value"),
        Input("s-force-start", "value"),
        Input("s-force-end", "value"),
        Input("gust-enabled", "value"),
        Input("s-gust", "value"),
        Input("s-gust-tau", "value"),
        Input("s-gust-start", "value"),
        Input("s-gust-end", "value"),
        Input("s-delay", "value"),
        Input("s-noise", "value"),
        Input("s-gain-kp", "value"),
        Input("s-gain-ki", "value"),
        Input("s-gain-kd", "value"),
    )
    def render_comparison(
        path_key: str,
        target_speed: Any,
        mass: Any,
        friction: Any,
        actuator: Any,
        mass_values: list[str],
        mass_target: Any,
        mass_time: Any,
        force_values: list[str],
        force: Any,
        force_start: Any,
        force_end: Any,
        gust_values: list[str],
        gust: Any,
        gust_tau: Any,
        gust_start: Any,
        gust_end: Any,
        delay_ms: Any,
        noise_mm: Any,
        kp: Any,
        ki: Any,
        kd: Any,
    ):
        try:
            result = runner.compare(
                path_key,
                target_speed,
                mass,
                friction,
                actuator,
                "on" in (mass_values or []),
                mass_target,
                mass_time,
                "on" in (force_values or []),
                force,
                force_start,
                force_end,
                "on" in (gust_values or []),
                gust,
                gust_tau,
                gust_start,
                gust_end,
                delay_ms,
                noise_mm,
                kp,
                ki,
                kd,
            )
            return comparison_view(result, runner)
        except ValueError as error:
            return html.Div(str(error), className="message error")
        except Exception:
            return html.Div("The comparison could not be completed.", className="message error")

    @app.callback(
        Output("replay-status", "children"),
        Input("replay-fixed", "n_clicks"),
        Input({"role": "replay-model", "index": ALL}, "n_clicks"),
        State("path-select", "value"),
        State("speed-select", "value"),
        State("s-mass", "value"),
        State("s-friction", "value"),
        State("s-actuator", "value"),
        State("s-delay", "value"),
        State("s-noise", "value"),
        State("s-gain-kp", "value"),
        State("s-gain-ki", "value"),
        State("s-gain-kd", "value"),
        State("gust-enabled", "value"),
        State("s-gust", "value"),
        State("s-gust-tau", "value"),
        State("s-gust-start", "value"),
        State("s-gust-end", "value"),
        prevent_initial_call=True,
    )
    def launch_replay(
        _fixed_clicks, _model_clicks, path_key, speed, mass, friction, actuator,
        delay_ms, noise_mm, kp, ki, kd, gust_values, gust, gust_tau,
        gust_start, gust_end,
    ):
        triggered = ctx.triggered_id
        entry = None
        if isinstance(triggered, dict) and triggered.get("role") == "replay-model":
            index = int(triggered["index"])
            if index >= len(runner.models):
                return "That model is no longer loaded."
            entry = runner.models[index]
        controller = "fixed" if entry is None else "ppo"
        command = [
            sys.executable, str(project / "replay.py"),
            "--controller", controller,
            "--path", str(path_key),
            "--speed", str(float(speed)),
            "--mass", str(float(mass)),
            "--friction", str(float(friction)),
            "--actuator", str(float(actuator)),
            "--delay-ms", str(float(delay_ms)),
            "--noise-mm", str(float(noise_mm)),
            "--gains", f"{float(kp)},{float(ki)},{float(kd)}",
            "--calibration", str(runner.calibration_path),
        ]
        if entry is not None:
            command += ["--model", str(entry["path"])]
        if "on" in (gust_values or []):
            command += [
                "--gust", str(float(gust)),
                "--gust-tau", str(float(gust_tau)),
                "--gust-start", str(float(gust_start)),
                "--gust-end", str(float(gust_end)),
            ]
        try:
            subprocess.Popen(command, cwd=str(project))
        except OSError as error:
            return f"Could not start the viewer: {error}"
        return (
            f"Launched the {entry['label'] if entry else 'Fixed PID'} replay "
            "in a separate window. Close that window when you are done; the "
            "episode metrics are printed in its console so you can check them "
            "against the panel."
        )

    @app.server.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        metavar="[LABEL=]PATH",
        help=(
            "PPO artifact to show as a panel. Repeat for several, e.g. "
            "--model \"blind=runs/rq1_blind/seed21/best_model.zip\" "
            "--model \"context=runs/rq2_context/seed42/best_model.zip\". "
            "Each model's observation arms are read from the arms.json beside "
            "it, so blind and plant-context models can be shown together. "
            "Omitted, the dashboard looks for the known run directories and "
            "falls back to the Fixed PID panel alone."
        ),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help=(
            "Gain box and baseline gains. Defaults to the newest of "
            + ", ".join(
                c.name for c in DirectComparisonRunner.CALIBRATION_CANDIDATES
            )
            + ". Must be the artifact the model was TRAINED against: the action "
            "vector is interpreted through low/base/high."
        ),
    )
    parser.add_argument("--port", type=int, default=8060)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        model_path=args.model, calibration_path=args.calibration
    )
    app.run(debug=False, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
