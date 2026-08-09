"""Watch one dashboard episode in the MuJoCo viewer.

Runs as its own process. MuJoCo's viewer wants the main thread and would fight
the Dash server, so the dashboard launches this rather than embedding it.

The episode is re-simulated here rather than shipped over as data, which is safe
because the environment is deterministic given the same options and seeds -- the
same scenario always produces the same trace. The metrics are printed on startup
so you can check them against the dashboard panel rather than taking that on
trust.

    python replay.py --path zigzag --speed 0.7 --gust 8 --controller ppo

Everything drawn is cosmetic and never touches physics: the reference path as
flat markers, and a live arrow over the car showing which way the wind is
pushing and how hard.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from calibration import GainCalibration
import config
from core import paths
from core.dynamics import car_id, fresh_model
from env import PathFollowingEnv
from rollout import fixed_policy, run_episode

MARKER_STRIDE = 10
MARKER_SIZE = (0.02, 0.02, 0.001)
MARKER_RGBA = (0.55, 0.62, 0.75, 0.9)
WIND_RGBA = (0.56, 0.72, 1.0, 0.95)
WIND_WIDTH = 0.012
# Metres of arrow per newton of wind. Chosen so the 6 N evaluation gust reads as
# roughly half a car length: legible without hiding the car underneath it.
WIND_SCALE = 0.05
WIND_HEIGHT = 0.16
END_PAUSE_S = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("fixed", "ppo"), default="fixed")
    parser.add_argument("--path", dest="path_key", default="arc")
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--mass", type=float, default=config.NOMINAL_MASS_KG)
    parser.add_argument("--friction", type=float, default=config.NOMINAL_FRICTION)
    parser.add_argument("--actuator", type=float, default=config.NOMINAL_ACTUATOR)
    parser.add_argument("--delay-ms", type=float, default=0.0)
    parser.add_argument("--noise-mm", type=float, default=0.0)
    parser.add_argument("--gust", type=float, default=0.0, help="0 disables wind")
    parser.add_argument("--gust-tau", type=float, default=config.EVAL_GUST_TAU_S)
    parser.add_argument("--gust-start", type=float, default=0.0)
    parser.add_argument("--gust-end", type=float, default=1e9)
    parser.add_argument("--gains", type=str, default=None, help="Kp,Ki,Kd")
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--playback", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    return parser.parse_args()


def resolve_calibration(project: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return project / explicit
    for name in ("calibration_v7.json", "calibration_v5.json", "calibration_twosided.json"):
        candidate = project / "runs" / name
        if candidate.exists():
            return candidate
    raise SystemExit("No calibration artifact found; run calibrate_pid.py first.")


def simulate(args: argparse.Namespace, project: Path):
    """Run the episode once, capturing a frame per control step (50 fps)."""
    calibration = GainCalibration.load(resolve_calibration(project, args.calibration))
    gains = None
    policy = None
    if args.controller == "fixed":
        gains = calibration.base.copy()
        if args.gains:
            values = [float(v) for v in args.gains.replace(",", " ").split()]
            if len(values) != 3:
                raise SystemExit("--gains needs exactly Kp,Ki,Kd")
            gains = np.asarray(values, dtype=np.float64)
    else:
        from stable_baselines3 import PPO

        model_path = project / (args.model or Path("runs") / "v7_seed7" / "final_model.zip")
        if not model_path.exists():
            raise SystemExit(f"No PPO model at {model_path}")
        model = PPO.load(model_path, device="cpu")

        def policy(observation: np.ndarray) -> np.ndarray:
            action, _ = model.predict(observation, deterministic=True)
            return np.asarray(action, dtype=np.float32)

    disturbances = [{"kind": "none", "start_s": 0.0, "end_s": None, "amount": 0.0}]
    if args.gust > 0.0:
        disturbances = [{
            "kind": "force_gust",
            "start_s": args.gust_start,
            "end_s": None if args.gust_end >= 1e8 else args.gust_end,
            "amount": args.gust,
            "tau_s": args.gust_tau,
            "seed": config.SEED,
        }]

    options = {
        "path_key": args.path_key,
        "v_target": args.speed,
        "mass": args.mass,
        "friction": args.friction,
        "actuator": args.actuator,
        "actuator_delay_s": 0.001 * args.delay_ms,
        "sensor_noise_m": 0.001 * args.noise_mm,
        "noise_seed": config.SEED,
        "stage": 3,
        "disturbances": disturbances,
    }

    env = PathFollowingEnv(
        calibration=calibration,
        training=False,
        path_keys=(args.path_key,),
        fixed_gains=gains,
    )
    frames: list[np.ndarray] = []
    winds: list[np.ndarray] = []

    # run_episode drives the loop; this samples the simulator state after each
    # control step, which is the same 50 Hz cadence the sandbox viewer used.
    def recording_policy(observation: np.ndarray) -> np.ndarray:
        frames.append(env.data.qpos.copy())
        winds.append(np.asarray(env.data.xfrc_applied[car_id(), 0:2]).copy())
        return (fixed_policy if gains is not None else policy)(observation)

    try:
        metrics, _trace = run_episode(
            env, recording_policy, seed=config.SEED, options=options
        )
        frames.append(env.data.qpos.copy())
        winds.append(np.asarray(env.data.xfrc_applied[car_id(), 0:2]).copy())
    finally:
        env.close()
    return metrics, np.array(frames), np.array(winds), options, calibration, gains


def draw_path(viewer, points: np.ndarray) -> int:
    marks = points[::MARKER_STRIDE]
    room = viewer.user_scn.maxgeom - 1  # keep one slot for the wind arrow
    if len(marks) > room:
        marks = marks[:: int(np.ceil(len(marks) / room))]
    for index, (x, y) in enumerate(marks):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[index],
            mujoco.mjtGeom.mjGEOM_BOX,
            np.array(MARKER_SIZE),
            np.array([x, y, 0.002]),
            np.eye(3).flatten(),
            np.array(MARKER_RGBA, dtype=np.float32),
        )
    # Set here rather than by the caller, so the scene is consistent the moment
    # this returns. The count doubles as the slot index for the wind arrow.
    viewer.user_scn.ngeom = len(marks)
    return len(marks)


def draw_wind(viewer, slot: int, position: np.ndarray, force: np.ndarray) -> None:
    """An arrow over the car pointing downwind, its length the gust strength."""
    magnitude = float(np.hypot(*force))
    if magnitude < 1e-6:
        viewer.user_scn.ngeom = slot
        return
    start = np.array([position[0], position[1], WIND_HEIGHT])
    end = start + np.array([force[0], force[1], 0.0]) * WIND_SCALE
    geom = viewer.user_scn.geoms[slot]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).flatten(),
        np.array(WIND_RGBA, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, WIND_WIDTH, start, end)
    viewer.user_scn.ngeom = slot + 1


def main() -> None:
    project = Path(__file__).resolve().parent
    args = parse_args()
    if args.path_key not in paths.CATALOGUE:
        raise SystemExit(f"unknown path: {args.path_key}")

    metrics, frames, winds, options, _calibration, gains = simulate(args, project)
    label = "Fixed PID" if args.controller == "fixed" else "PPO"
    if gains is not None:
        label += f"  Kp {gains[0]:.2f} Ki {gains[1]:.2f} Kd {gains[2]:.2f}"
    print(f"\n{label}   {args.path_key} @ {args.speed} m/s")
    print(f"  finished     : {metrics['finished']}  ({metrics['failure_reason'] or 'ok'})")
    print(f"  mean distance: {metrics['mean_distance_m']*100:.2f} cm")
    print(f"  max distance : {metrics['max_distance_m']*100:.2f} cm")
    print(f"  duration     : {metrics['duration_s']:.2f} s over {len(frames)} frames")
    print("\nClose the viewer window to exit.")

    model = fresh_model(options["mass"], options["friction"], options["actuator"])
    data = mujoco.MjData(model)
    points = paths.get(args.path_key)["pts"]
    frame_dt = config.FRAME_SKIP * float(model.opt.timestep)

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        wind_slot = draw_path(viewer, points)

        span = float(np.max(np.ptp(points, axis=0)))
        viewer.cam.lookat[:] = (*points.mean(axis=0), 0.0)
        viewer.cam.distance = max(2.5, span * 1.6)
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -75.0

        while viewer.is_running():
            start_wall = time.perf_counter()
            for index, frame in enumerate(frames):
                if not viewer.is_running():
                    return
                data.qpos[:] = frame
                mujoco.mj_forward(model, data)
                draw_wind(viewer, wind_slot, frame[0:2], winds[index])
                viewer.sync()
                target = start_wall + (index + 1) * frame_dt / max(args.playback, 0.05)
                lag = target - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
            if not args.loop:
                break
            time.sleep(END_PAUSE_S)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
