"""Gymnasium environment for causal PPO steering-PID gain scheduling."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from calibration import GainCalibration
import config
from core import paths
from core.dynamics import (
    CORRIDOR_M,
    DRIVE_SIGNS,
    DT,
    RUNAWAY_M,
    TrackState,
    allocate_steering_priority,
    car_id,
    forward_speed,
    fresh_model,
    set_mass,
    track_error,
    yaw_of,
)
from core.pid import ConditionalPIDController


FRAME_FIELDS = (
    "e_ct",
    "e_ct_rate",
    "steer_integral",
    "e_theta",
    "speed",
    "yaw_rate",
    "omega_requested",
    "omega_applied",
    "v_applied",
    "previous_action_kp",
    "previous_action_ki",
    "previous_action_kd",
    "v_target",
    "wheel_utilization",
    "pid_saturated",
    "allocation_limited",
    "steer_integrator_hold",
)


def tracking_cost_of(distances: Any) -> float:
    """Mean saturating cost of being this far off the path, bounded in [0, 1).

    Quadratic near zero, so sub-centimetre differences between controllers are
    resolved; flat well before the corridor edge, so an episode spent far off
    path is merely bad rather than arithmetically overwhelming. See
    `config.TRACKING_SCALE_M` for why both halves of that matter.
    """
    squared = (np.asarray(distances, dtype=np.float64) / config.TRACKING_SCALE_M) ** 2
    return float(np.mean(squared / (1.0 + squared)))


class PathFollowingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Blind gain scheduler with a 10-frame causal observation history."""

    metadata = {"render_modes": [], "render_fps": 50}

    def __init__(
        self,
        *,
        calibration: GainCalibration | None = None,
        training: bool = True,
        path_keys: tuple[str, ...] | list[str] | None = None,
        fixed_gains: np.ndarray | None = None,
        gusts: bool = True,
        delay: bool = True,
        noise: bool = True,
    ) -> None:
        super().__init__()
        self.calibration = calibration or GainCalibration.development_default()
        self.training = bool(training)
        # The no-gust control run. Comparing a gust-trained agent against one
        # trained under a different curriculum entirely proves nothing, so the
        # control has to differ in this one variable and nothing else.
        self.gusts = bool(gusts)
        # Same contract as `gusts`: each of these switches off exactly one plant
        # imperfection so an ablation differs in one variable. Both are ignored
        # outside training-stage randomisation -- evaluation always takes its
        # values from reset options, never from the curriculum.
        self.delay = bool(delay)
        self.noise = bool(noise)
        self.path_keys = tuple(path_keys or config.TRAIN_PATHS)
        unknown = set(self.path_keys) - set(paths.CATALOGUE)
        if unknown:
            raise ValueError(f"unknown path keys: {sorted(unknown)}")
        self.fixed_gains = (
            None if fixed_gains is None else np.asarray(fixed_gains, dtype=np.float64)
        )
        if self.fixed_gains is not None and self.fixed_gains.shape != (3,):
            raise ValueError("fixed_gains must contain [Kp, Ki, Kd]")

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        obs_size = len(FRAME_FIELDS) * config.HISTORY_LENGTH
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(obs_size,), dtype=np.float32
        )
        self.render_mode = None
        self.training_progress = 0.0
        self._history: deque[np.ndarray] = deque(maxlen=config.HISTORY_LENGTH)
        self._needs_reset = True

    @property
    def control_dt(self) -> float:
        return DT * config.FRAME_SKIP

    def set_training_progress(self, progress: float) -> None:
        self.training_progress = float(np.clip(progress, 0.0, 1.0))

    def _curriculum_stage(self) -> int:
        if self.training_progress < config.CURRICULUM_NOMINAL_END:
            return 1
        if self.training_progress < config.CURRICULUM_RANDOMIZED_END:
            return 2
        return 3

    def _sample_episode(self, options: dict[str, Any]) -> dict[str, Any]:
        stage = int(options.get("stage", self._curriculum_stage()))
        if self.training and stage == 1:
            available_paths = tuple(k for k in self.path_keys if k in {"arc", "scurve", "uturn"})
            available_paths = available_paths or self.path_keys
            speeds = (0.3, 0.5)
            mass = config.NOMINAL_MASS_KG
            friction = config.NOMINAL_FRICTION
            actuator = config.NOMINAL_ACTUATOR
            delay_s = 0.0
            noise_m = 0.0
        else:
            available_paths = self.path_keys
            speeds = config.SPEED_TARGETS
            if self.training:
                mass = self.np_random.uniform(*config.MASS_RANGE_KG)
                friction = self.np_random.uniform(*config.FRICTION_RANGE)
                actuator = self.np_random.uniform(*config.ACTUATOR_RANGE)
                # Drawn unconditionally so that turning one imperfection off
                # does not shift the other's draw sequence: an ablation has to
                # differ in the plant, not in which random numbers it consumed.
                drawn_delay = self.np_random.uniform(*config.ACTUATOR_DELAY_RANGE_S)
                drawn_noise = self.np_random.uniform(*config.SENSOR_NOISE_RANGE_M)
                delay_s = drawn_delay if self.delay else 0.0
                noise_m = drawn_noise if self.noise else 0.0
            else:
                mass = config.NOMINAL_MASS_KG
                friction = config.NOMINAL_FRICTION
                actuator = config.NOMINAL_ACTUATOR
                delay_s = 0.0
                noise_m = 0.0

        choice = {
            "stage": stage,
            "path_key": str(self.np_random.choice(available_paths)),
            "v_target": float(self.np_random.choice(speeds)),
            "mass": float(mass),
            "friction": float(friction),
            "actuator": float(actuator),
            "actuator_delay_s": float(delay_s),
            "sensor_noise_m": float(noise_m),
            # The noise stream gets its own seed for the same reason gusts do:
            # the Fixed PID and the PPO policy must meet the SAME sensor trace,
            # and drawing it from self.np_random would couple the sequence to
            # how many other draws each run happened to make.
            "noise_seed": int(self.np_random.integers(0, 2**31 - 1)),
        }
        for key in (
            "path_key",
            "v_target",
            "mass",
            "friction",
            "actuator",
            "actuator_delay_s",
            "sensor_noise_m",
            "noise_seed",
        ):
            if key in options:
                choice[key] = options[key]
        return choice

    def _sample_disturbance(self, stage: int, max_time_s: float) -> dict[str, Any]:
        off = {"kind": "none", "start_s": 0.0, "end_s": 0.0, "amount": 0.0}
        if not self.training or stage < 3:
            return off
        start = float(self.np_random.uniform(0.30, 0.55) * max_time_s)
        end = min(max_time_s, start + float(self.np_random.uniform(0.15, 0.30) * max_time_s))
        # Two-way split for the control run, three-way when gusts are enabled.
        draw = self.np_random.random() * (1.0 if self.gusts else 2.0 / 3.0)
        if draw < 1.0 / 3.0:
            target_mass = float(self.np_random.uniform(*config.MASS_RANGE_KG))
            return {
                "kind": "mass",
                "start_s": start,
                "end_s": end,
                "amount": target_mass - self.mass,
            }
        if draw < 2.0 / 3.0:
            amount = float(self.np_random.uniform(-5.0, 5.0))
            if abs(amount) < 1.0:
                amount = math.copysign(1.0, amount or 1.0)
            return {"kind": "force", "start_s": start, "end_s": end, "amount": amount}
        # Wind runs for most of the episode rather than as a brief pulse. A gust
        # confined to the 15-30% window the other two events use would leave the
        # agent almost no time under wind to react to, and adapting is the whole
        # point of putting it in the curriculum.
        gust_start = float(self.np_random.uniform(0.05, 0.35) * max_time_s)
        return {
            "kind": "force_gust",
            "start_s": gust_start,
            "end_s": max_time_s,
            "amount": float(self.np_random.uniform(*config.GUST_SIGMA_RANGE_N)),
            "tau_s": float(self.np_random.uniform(*config.GUST_TAU_RANGE_S)),
            # A fresh realisation per episode. Reusing one seed would let the
            # agent memorise a single wind trace instead of learning to handle
            # wind. Drawn from np_random, so the episode stays reproducible.
            "seed": int(self.np_random.integers(0, 2**31 - 1)),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = dict(options or {})
        episode = self._sample_episode(options)
        self.stage = int(episode["stage"])
        self.path_key = str(episode["path_key"])
        self.path = paths.get(self.path_key)
        self.v_target = float(episode["v_target"])
        self.mass = float(episode["mass"])
        self.friction = float(episode["friction"])
        self.actuator = float(episode["actuator"])
        self.actuator_delay_s = max(0.0, float(episode["actuator_delay_s"]))
        self.sensor_noise_m = max(0.0, float(episode["sensor_noise_m"]))
        self.noise_seed = int(episode["noise_seed"])
        # Quantised to whole physics steps, because that is the resolution the
        # command buffer has. The stored seconds value is the rounded one, so
        # traces and manifests report what was actually applied.
        self.delay_steps = int(round(self.actuator_delay_s / DT))
        self.actuator_delay_s = self.delay_steps * DT
        self._noise_rng = np.random.default_rng(self.noise_seed)
        self._command_queue: deque[tuple[float, float]] = deque(
            [(0.0, 0.0)] * self.delay_steps, maxlen=self.delay_steps + 1
        )

        self.model = fresh_model(self.mass, self.friction, self.actuator)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[0:3] = (0.0, 0.0, 0.03)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, self.data)

        self.steer_pid = ConditionalPIDController(
            kp=float(self.calibration.base[0]),
            ki=float(self.calibration.base[1]),
            kd=float(self.calibration.base[2]),
            setpoint=0.0,
            output_limits=(-1.0, 1.0),
        )
        speed_kp, speed_ki, speed_kd = config.SPEED_PID_GAINS
        self.speed_pid = ConditionalPIDController(
            kp=float(speed_kp),
            ki=float(speed_ki),
            kd=float(speed_kd),
            setpoint=self.v_target,
            output_limits=(-1.0, 1.0),
        )

        ideal_time = float(self.path["length"]) / self.v_target
        max_time = config.TIME_LIMIT_FACTOR * ideal_time + config.TIME_LIMIT_MARGIN_S
        self.max_episode_steps = int(math.ceil(max_time / self.control_dt))
        if "max_episode_steps" in options:
            self.max_episode_steps = int(options["max_episode_steps"])
        self.disturbance = self._sample_disturbance(self.stage, max_time)
        if "disturbance" in options:
            self.disturbance = deepcopy(options["disturbance"])
        configured_disturbances = options.get("disturbances")
        if configured_disturbances is None:
            self.disturbances = [self.disturbance]
        else:
            if not isinstance(configured_disturbances, list) or not configured_disturbances:
                raise ValueError("disturbances must be a non-empty list")
            self.disturbances = deepcopy(configured_disturbances)
            self.disturbance = self.disturbances[0]

        self.path_index = 0
        self.elapsed_steps = 0
        self.physics_steps = 0
        self.time_s = 0.0
        self.previous_action = np.zeros(3, dtype=np.float64)
        self.target_gains = self.calibration.base.copy()
        self.applied_gains = self.calibration.base.copy()
        self._mass_event_active = False
        self._reset_gusts()
        self._trace: list[dict[str, Any]] = []
        self._distance_sum = 0.0
        self._distance_count = 0
        self._max_distance = 0.0
        self._itae = 0.0
        self._saturation_count = 0
        self._allocation_count = 0
        self._gain_change_sum = 0.0
        self._last_frame_e_ct = 0.0
        self._needs_reset = False

        state = self._state()
        measured_e_ct = self._measure(state.e_ct)
        self._last_frame_e_ct = measured_e_ct
        zero_control = {
            "omega_requested": 0.0,
            "omega_applied": 0.0,
            "v_applied": 0.0,
            "u_left_applied": 0.0,
            "u_right_applied": 0.0,
            "wheel_utilization": 0.0,
            "pid_saturated": False,
            "allocation_limited": False,
            "steer_integrator_hold": False,
        }
        frame = self._frame(state, measured_e_ct, 0.0, zero_control)
        self._history.clear()
        for _ in range(config.HISTORY_LENGTH):
            self._history.append(frame.copy())
        return self._observation(), self._base_info(state)

    def _measure(self, e_ct_true: float) -> float:
        """The cross-track error the CONTROLLER sees, which is not the truth.

        Everything the study scores -- reward, metrics, corridor termination --
        keeps using the true state. Only the steering PID and the observation
        consume this. Mixing the two would score the controller on hallucinated
        performance.

        Only `e_ct` is corrupted. Heading and speed are separate sensors and the
        steering PID does not read them, so noising them would perturb the
        observation without touching the gain tradeoff this models.
        """
        if self.sensor_noise_m <= 0.0:
            return float(e_ct_true)
        return float(e_ct_true + self._noise_rng.normal(0.0, self.sensor_noise_m))

    def _delayed_command(self, u_left: float, u_right: float) -> tuple[float, float]:
        """Return the wheel command issued `delay_steps` physics steps ago.

        A pure transport delay on the actuator channel. The PID's own
        anti-windup still sees the freshly allocated command, which is correct:
        allocation feasibility is a property of the request, and the real
        actuator has no way to tell the controller that its command is queued.
        """
        self._command_queue.append((float(u_left), float(u_right)))
        return self._command_queue.popleft()

    def _state(self) -> TrackState:
        car_xy = np.asarray(self.data.qpos[0:2], dtype=np.float64)
        yaw = yaw_of(self.data.qpos[3:7])
        state = track_error(self.path, car_xy, yaw, self.path_index)
        self.path_index = state.index
        return state

    def _event_active(self, event: dict[str, Any]) -> bool:
        if event["kind"] == "none":
            return False
        end_s = event.get("end_s")
        return self.time_s >= float(event["start_s"]) and (
            end_s is None or self.time_s < float(end_s)
        )

    def _disturbance_active(self) -> bool:
        return any(self._event_active(event) for event in self.disturbances)

    def _disturbance_kind(self) -> str:
        kinds = [event["kind"] for event in self.disturbances if event["kind"] != "none"]
        return "+".join(kinds) if kinds else "none"

    def _disturbance_value(self) -> float:
        if len(self.disturbances) == 1:
            return float(self.disturbances[0]["amount"])
        return 0.0

    def _reset_gusts(self) -> None:
        """Give every gust event its own generator, seeded from the event.

        Deliberately NOT drawn from ``self.np_random``: the whole point of the
        comparison is that the Fixed PID and the PPO policy meet the *same* wind.
        Sharing the episode generator would couple the gust sequence to how many
        other draws each run happened to make, so the two controllers would be
        fighting different weather and the comparison would mean nothing.
        """
        self._gust_rng: dict[int, np.random.Generator] = {}
        self._gust_force: dict[int, np.ndarray] = {}
        for index, event in enumerate(self.disturbances):
            if event["kind"] != "force_gust":
                continue
            seed = int(event.get("seed", config.SEED))
            self._gust_rng[index] = np.random.default_rng(seed)
            # Starts calm and builds over roughly one correlation time, rather
            # than jolting the car with a stationary draw the instant it arms.
            self._gust_force[index] = np.zeros(2, dtype=np.float64)

    def _advance_gust(self, index: int, event: dict[str, Any]) -> np.ndarray:
        """One exact Ornstein-Uhlenbeck step, per world axis.

        ``F <- F*decay + sigma*sqrt(1 - decay^2)*N(0, 1)`` is the exact update
        for the OU process sampled at DT, not an Euler approximation, so the
        stationary spread stays at ``sigma`` no matter what DT or tau are set
        to. Two independent axes means the magnitude wanders (Rayleigh about
        ``sigma``) and so does the bearing -- a wind that veers, not a square
        pulse that flips.
        """
        sigma = float(event["amount"])
        tau = max(float(event.get("tau_s", 1.0)), DT)
        decay = math.exp(-DT / tau)
        step = self._gust_rng[index].normal(size=2)
        force = self._gust_force[index] * decay + sigma * math.sqrt(1.0 - decay**2) * step
        self._gust_force[index] = force
        return force

    def _apply_disturbance(self) -> None:
        mass_event = next(
            (event for event in self.disturbances if event["kind"] in {"mass", "mass_step"}),
            None,
        )
        mass_active = mass_event is not None and self._event_active(mass_event)
        if mass_event is not None and mass_active != self._mass_event_active:
            amount = float(mass_event["amount"])
            event_mass = amount if mass_event["kind"] == "mass_step" else self.mass + amount
            set_mass(self.model, event_mass if mass_active else self.mass)
            self._mass_event_active = mass_active
        self.data.xfrc_applied[car_id(), 0:2] = 0.0
        for index, event in enumerate(self.disturbances):
            if not self._event_active(event):
                continue
            if event["kind"] == "force":
                self.data.xfrc_applied[car_id(), 0] += float(event["amount"])
            elif event["kind"] == "force_pulse":
                yaw = yaw_of(self.data.qpos[3:7])
                force = float(event["amount"])
                self.data.xfrc_applied[car_id(), 0] += -math.sin(yaw) * force
                self.data.xfrc_applied[car_id(), 1] += math.cos(yaw) * force
            elif event["kind"] == "force_gust":
                self.data.xfrc_applied[car_id(), 0:2] += self._advance_gust(index, event)

    def _frame(
        self,
        state: TrackState,
        e_ct_measured: float,
        e_ct_rate: float,
        control: dict[str, Any],
    ) -> np.ndarray:
        yaw = yaw_of(self.data.qpos[3:7])
        speed = forward_speed(self.data, yaw)
        values = np.array(
            [
                e_ct_measured / config.E_CT_SCALE_M,
                e_ct_rate / config.E_CT_RATE_SCALE_MPS,
                self.steer_pid._integral / config.INTEGRAL_SCALE_MS,
                state.e_theta / np.pi,
                speed / config.SPEED_SCALE_MPS,
                float(self.data.qvel[5]) / config.YAW_RATE_SCALE_RADPS,
                control["omega_requested"],
                control["omega_applied"],
                control["v_applied"],
                *self.previous_action,
                self.v_target / config.MAX_TARGET_SPEED_MPS,
                control["wheel_utilization"],
                float(control["pid_saturated"]),
                float(control["allocation_limited"]),
                float(control["steer_integrator_hold"]),
            ],
            dtype=np.float32,
        )
        return np.clip(values, -1.0, 1.0)

    def _observation(self) -> np.ndarray:
        return np.concatenate(tuple(self._history)).astype(np.float32, copy=False)

    def _base_info(self, state: TrackState) -> dict[str, Any]:
        return {
            "path_key": self.path_key,
            "v_target": self.v_target,
            "mass": self.mass,
            "friction": self.friction,
            "actuator": self.actuator,
            "curriculum_stage": self.stage,
            "actuator_delay_s": self.actuator_delay_s,
            "sensor_noise_m": self.sensor_noise_m,
            "progress_m": state.progress_m,
            "distance_m": state.distance_m,
            "disturbance": deepcopy(self.disturbance),
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise RuntimeError("reset() must be called before step() or after an episode ends")
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if action.shape != (3,):
            raise ValueError("action must contain three normalized gain targets")
        previous_action = self.previous_action.copy()
        self.target_gains = (
            self.fixed_gains.copy()
            if self.fixed_gains is not None
            else self.calibration.gains_from_action(action)
        )
        self.applied_gains = self.target_gains.copy()
        self.steer_pid.kp, self.steer_pid.ki, self.steer_pid.kd = self.applied_gains

        state_before = self._state()
        start_progress = state_before.progress_m
        substep_distances: list[float] = []
        terminated = False
        failure_reason = ""
        finished = False
        control: dict[str, Any] = {}

        for _ in range(config.FRAME_SKIP):
            self._apply_disturbance()
            state_for_control = self._state()
            yaw = yaw_of(self.data.qpos[3:7])
            speed = forward_speed(self.data, yaw)
            e_ct_measured = self._measure(state_for_control.e_ct)
            omega_requested = float(self.steer_pid.update(e_ct_measured, DT))
            v_requested = float(self.speed_pid.update(speed, DT))
            (
                u_left_requested,
                u_right_requested,
                u_left,
                u_right,
                v_applied,
                omega_applied,
            ) = allocate_steering_priority(v_requested, omega_requested)
            self.steer_pid.apply_output_feedback(omega_applied)
            self.speed_pid.apply_output_feedback(v_applied)
            u_left_applied, u_right_applied = self._delayed_command(u_left, u_right)
            self.data.ctrl[0] = DRIVE_SIGNS[0] * u_left_applied
            self.data.ctrl[1] = DRIVE_SIGNS[1] * u_right_applied
            mujoco.mj_step(self.model, self.data)
            self.physics_steps += 1
            self.time_s = self.physics_steps * DT

            state = self._state()
            substep_distances.append(state.distance_m)
            self._distance_sum += state.distance_m
            self._distance_count += 1
            self._max_distance = max(self._max_distance, state.distance_m)
            self._itae += self.time_s * state.distance_m * DT
            pid_saturated = abs(self.steer_pid.last_unclamped_output) > 1.0 + 1e-12
            allocation_limited = not (
                np.isclose(v_requested, v_applied)
                and np.isclose(omega_requested, omega_applied)
            )
            wheel_utilization = max(abs(u_left), abs(u_right))
            self._saturation_count += int(wheel_utilization > 0.99)
            self._allocation_count += int(allocation_limited)
            control = {
                "omega_unclamped": self.steer_pid.last_unclamped_output,
                "omega_requested": omega_requested,
                "omega_applied": omega_applied,
                "v_requested": v_requested,
                "v_applied": v_applied,
                "u_left_requested": u_left_requested,
                "u_right_requested": u_right_requested,
                "u_left": u_left,
                "u_right": u_right,
                # What the wheels actually received this substep, which under a
                # non-zero dead time is an older command than `u_left`/`u_right`.
                # Utilization and the saturation counters deliberately stay on
                # the REQUEST: those describe the controller's demand, and are
                # observable to it, whereas the queued command is not.
                "u_left_applied": u_left_applied,
                "u_right_applied": u_right_applied,
                "wheel_utilization": wheel_utilization,
                "pid_saturated": pid_saturated,
                "allocation_limited": allocation_limited,
                "steer_integrator_hold": (
                    self.steer_pid.last_internal_hold
                    or self.steer_pid.last_downstream_hold
                ),
            }

            car_xy = np.asarray(self.data.qpos[0:2])
            if not np.isfinite(self.data.qpos).all() or np.abs(car_xy).max() > RUNAWAY_M:
                terminated, failure_reason = True, "invalid_state"
            elif abs(state.e_ct) > CORRIDOR_M or state.distance_m > CORRIDOR_M:
                terminated, failure_reason = True, "corridor_exit"
            elif state.index >= len(self.path["pts"]) - 1:
                terminated, finished = True, True
            if terminated:
                break

        self.elapsed_steps += 1
        truncated = self.elapsed_steps >= self.max_episode_steps and not terminated
        if truncated:
            failure_reason = "time_limit"

        elapsed_control_time = max(len(substep_distances), 1) * DT
        # One fresh sensor read at the observation instant, and the rate is the
        # difference of two MEASURED values -- that is what a real derivative
        # term and a real policy would have to work with. Differencing the true
        # state here would hand the agent a clean signal the PID never sees.
        measured_e_ct = self._measure(state.e_ct)
        e_ct_rate = (measured_e_ct - self._last_frame_e_ct) / elapsed_control_time
        self._last_frame_e_ct = measured_e_ct
        self.previous_action = action.copy()
        self._history.append(self._frame(state, measured_e_ct, e_ct_rate, control))

        progress_delta = max(0.0, state.progress_m - start_progress)
        tracking_cost = tracking_cost_of(substep_distances)
        smoothness_cost = float(np.sum((action - previous_action) ** 2))
        reward_progress = config.PROGRESS_WEIGHT * progress_delta
        reward_tracking = -config.TRACKING_WEIGHT * tracking_cost
        reward_smoothness = -config.ACTION_SMOOTHNESS_WEIGHT * smoothness_cost
        reward_terminal = (
            config.FINISH_BONUS
            if finished
            else -config.FAILURE_PENALTY if (terminated or truncated) else 0.0
        )
        reward = reward_progress + reward_tracking + reward_smoothness + reward_terminal
        self._gain_change_sum += float(np.linalg.norm(action - previous_action))

        info = self._base_info(state)
        info.update(
            {
                "finished": finished,
                "failure_reason": failure_reason,
                "e_ct_measured": measured_e_ct,
                "target_gains": self.target_gains.copy(),
                "applied_gains": self.applied_gains.copy(),
                "action": action.copy(),
                "reward_progress": reward_progress,
                "reward_tracking": reward_tracking,
                "reward_smoothness": reward_smoothness,
                "reward_terminal": reward_terminal,
                **control,
            }
        )
        self._append_trace(state, reward, info)

        if terminated or truncated:
            self._needs_reset = True
            info["episode_metrics"] = self.episode_metrics(
                finished=finished, failure_reason=failure_reason
            )
        return self._observation(), float(reward), terminated, truncated, info

    def _append_trace(self, state: TrackState, reward: float, info: dict[str, Any]) -> None:
        yaw = yaw_of(self.data.qpos[3:7])
        row: dict[str, Any] = {
            "t": self.time_s,
            "x": float(self.data.qpos[0]),
            "y": float(self.data.qpos[1]),
            "yaw": yaw,
            "e_ct": state.e_ct,
            "e_theta": state.e_theta,
            "dist": state.distance_m,
            "progress_s": state.progress_m,
            "speed": forward_speed(self.data, yaw),
            "yaw_rate": float(self.data.qvel[5]),
            "reward": reward,
            "path_key": self.path_key,
            "v_target": self.v_target,
            "base_mass": self.mass,
            "mass": float(self.model.body_mass[car_id()]),
            "friction": self.friction,
            "actuator": self.actuator,
            "actuator_delay_s": self.actuator_delay_s,
            "sensor_noise_m": self.sensor_noise_m,
            "external_force_x": float(self.data.xfrc_applied[car_id(), 0]),
            "external_force_y": float(self.data.xfrc_applied[car_id(), 1]),
            "event_kind": self._disturbance_kind(),
            "event_value": self._disturbance_value(),
            "disturbance_active": self._disturbance_active(),
        }
        for name, value in zip(("kp", "ki", "kd"), self.applied_gains):
            row[name] = float(value)
        for name, value in zip(("action_kp", "action_ki", "action_kd"), self.previous_action):
            row[name] = float(value)
        for key in (
            "e_ct_measured",
            "omega_unclamped",
            "omega_requested",
            "omega_applied",
            "v_requested",
            "v_applied",
            "u_left",
            "u_right",
            "u_left_applied",
            "u_right_applied",
            "wheel_utilization",
            "pid_saturated",
            "allocation_limited",
            "steer_integrator_hold",
            "reward_progress",
            "reward_tracking",
            "reward_smoothness",
            "reward_terminal",
        ):
            row[key] = info[key]
        self._trace.append(row)

    def episode_metrics(self, *, finished: bool, failure_reason: str) -> dict[str, Any]:
        count = max(self._distance_count, 1)
        return {
            "finished": bool(finished),
            "failure_reason": failure_reason,
            "path_key": self.path_key,
            "v_target": self.v_target,
            "mass": self.mass,
            "friction": self.friction,
            "actuator": self.actuator,
            "actuator_delay_s": self.actuator_delay_s,
            "sensor_noise_m": self.sensor_noise_m,
            "duration_s": self.time_s,
            "progress_pct": 100.0 * float(self.path["s"][self.path_index]) / float(self.path["length"]),
            "mean_distance_m": self._distance_sum / count,
            "max_distance_m": self._max_distance,
            "itae": self._itae,
            "saturation_fraction": self._saturation_count / count,
            "allocation_limited_fraction": self._allocation_count / count,
            "mean_action_change": self._gain_change_sum / max(self.elapsed_steps, 1),
        }

    def get_trace(self) -> list[dict[str, Any]]:
        return deepcopy(self._trace)

    def close(self) -> None:
        self._needs_reset = True
