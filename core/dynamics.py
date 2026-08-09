"""Step-wise MuJoCo plant shared by fixed PID and PPO gain scheduling.

The formulas and sign conventions are copied from Path_Following_Sandbox/sim.py.
Keeping them here makes the Gymnasium environment self-contained while allowing
unit tests and calibration to exercise the exact same low-level controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


XML_PATH = str(Path(__file__).with_name("car_model.xml"))
DRIVE_SIGNS = (+1.0, +1.0)
CORRIDOR_M = 1.0
RUNAWAY_M = 20.0
SEARCH_W = 200
MIN_MASS_KG = 0.5

_REFERENCE_MODEL = mujoco.MjModel.from_xml_path(XML_PATH)
_CAR_ID = mujoco.mj_name2id(
    _REFERENCE_MODEL, mujoco.mjtObj.mjOBJ_BODY, "car"
)
_FLOOR_ID = mujoco.mj_name2id(
    _REFERENCE_MODEL, mujoco.mjtObj.mjOBJ_GEOM, "floor"
)
DT = float(_REFERENCE_MODEL.opt.timestep)
NOMINAL = {
    "mass": float(_REFERENCE_MODEL.body_mass[_CAR_ID]),
    "inertia": _REFERENCE_MODEL.body_inertia[_CAR_ID].copy(),
    "friction": float(_REFERENCE_MODEL.geom_friction[_FLOOR_ID, 0]),
    "gainprm": _REFERENCE_MODEL.actuator_gainprm.copy(),
    "actfrcrange": _REFERENCE_MODEL.jnt_actfrcrange.copy(),
}


@dataclass(frozen=True)
class TrackState:
    index: int
    e_ct: float
    e_theta: float
    progress_m: float
    distance_m: float


def fresh_model(mass: float, friction: float, actuator: float) -> mujoco.MjModel:
    """Load a fresh model and apply one episode's physical parameters."""
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    set_mass(model, mass)
    model.geom_friction[:, 0] = float(friction)
    model.actuator_gainprm[:, 0] = NOMINAL["gainprm"][:, 0] * float(actuator)
    model.jnt_actfrcrange[:] = NOMINAL["actfrcrange"] * float(actuator)
    return model


def set_mass(model: mujoco.MjModel, mass: float) -> None:
    """Set chassis mass and scale its inertia exactly as the sandbox does."""
    mass = max(float(mass), MIN_MASS_KG)
    model.body_mass[_CAR_ID] = mass
    model.body_inertia[_CAR_ID, :] = (
        NOMINAL["inertia"] * (mass / NOMINAL["mass"])
    )


def car_id() -> int:
    return _CAR_ID


def yaw_of(quat: np.ndarray) -> float:
    qw, qx, qy, qz = quat
    return float(
        np.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
    )


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def allocate_steering_priority(
    v_requested: float, omega_requested: float
) -> tuple[float, float, float, float, float, float]:
    """Preserve steering and reduce speed until both wheels are feasible."""
    v_requested = float(v_requested)
    omega_requested = float(omega_requested)
    u_left_requested = v_requested - omega_requested
    u_right_requested = v_requested + omega_requested
    omega_applied = float(np.clip(omega_requested, -1.0, 1.0))
    v_headroom = 1.0 - abs(omega_applied)
    v_applied = float(np.clip(v_requested, -v_headroom, v_headroom))
    u_left = v_applied - omega_applied
    u_right = v_applied + omega_applied
    return (
        u_left_requested,
        u_right_requested,
        u_left,
        u_right,
        v_applied,
        omega_applied,
    )


def track_error(
    path: dict, car_xy: np.ndarray, yaw: float, index: int
) -> TrackState:
    """Select the closest forward waypoint and calculate both error measures."""
    points, tangents = path["pts"], path["tangent"]
    end = min(index + SEARCH_W, len(points) - 1)
    distances_sq = np.sum((points[index : end + 1] - car_xy) ** 2, axis=1)
    index += int(np.argmin(distances_sq))
    tx, ty = tangents[index]
    dx, dy = car_xy - points[index]
    e_ct = float(tx * dy - ty * dx)
    e_theta = wrap_to_pi(float(np.arctan2(ty, tx)) - yaw)
    return TrackState(
        index=index,
        e_ct=e_ct,
        e_theta=e_theta,
        progress_m=float(path["s"][index]),
        distance_m=float(np.hypot(dx, dy)),
    )


def forward_speed(data: mujoco.MjData, yaw: float) -> float:
    return float(data.qvel[0]) * np.cos(yaw) + float(data.qvel[1]) * np.sin(yaw)

