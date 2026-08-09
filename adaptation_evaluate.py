"""Run deterministic paired fixed-PID and frozen-PPO adaptation scenarios."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import traceback
from typing import Any, Callable

import numpy as np
from stable_baselines3 import PPO

from adaptation_metrics import adaptation_metrics, aggregate_pairs, paired_row
from adaptation_scenarios import (
    Scenario,
    build_scenarios,
    smoke_scenarios,
    validate_scenario_set,
)
from calibration import GainCalibration
import config
from env import PathFollowingEnv
from rollout import fixed_policy, run_episode


CONTROLLERS = ("fixed", "ppo")
DEVELOPMENT_PATHS = ("arc", "slalom", "zigzag", "spiral", "figure8", "hairpin", "zigzag60")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_scenarios(path: Path) -> list[Scenario]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["scenarios"] if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("scenario file must contain a list or a scenarios list")
    scenarios = [Scenario.from_dict(record) for record in records]
    validate_scenario_set(scenarios)
    return scenarios


def package_versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "mujoco", "gymnasium", "stable-baselines3", "torch", "dash", "plotly"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def source_hashes(project_dir: Path) -> dict[str, str]:
    names = (
        "adaptation_scenarios.py",
        "adaptation_metrics.py",
        "adaptation_evaluate.py",
        "env.py",
        "rollout.py",
        "config.py",
        "core/dynamics.py",
        "core/pid.py",
        "core/paths.py",
        "core/car_model.xml",
    )
    return {name: sha256_file(project_dir / name) for name in names}


def default_run_id(scenarios: list[Scenario]) -> str:
    joined = "".join(scenario.fingerprint for scenario in scenarios)
    matrix_hash = hashlib.sha256(joined.encode("ascii")).hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{matrix_hash}"


def create_run_directory(root: Path, run_id: str) -> Path:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must be a short filesystem-safe name")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = (root / run_id).resolve()
    if run_dir.parent != root:
        raise ValueError("run directory escaped the configured root")
    run_dir.mkdir(exist_ok=False)
    for child in ("traces", "plots", "logs"):
        (run_dir / child).mkdir()
    return run_dir


def _policy(model: PPO) -> Callable[[np.ndarray], np.ndarray]:
    def predict(observation: np.ndarray) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)
    return predict


def run_controller(
    env: PathFollowingEnv,
    policy: Callable[[np.ndarray], np.ndarray],
    scenario: Scenario,
    controller: str,
    trace_path: Path,
) -> dict[str, Any]:
    metrics, trace = run_episode(
        env,
        policy,
        seed=scenario.seed,
        options=scenario.to_env_options(),
    )
    trace_rows = [
        {
            "scenario_id": scenario.scenario_id,
            "controller": controller,
            **row,
        }
        for row in trace
    ]
    write_csv(trace_path, trace_rows)
    saved_trace = read_csv(trace_path)
    windowed = adaptation_metrics(saved_trace, scenario)
    return {
        "scenario_id": scenario.scenario_id,
        "scenario_fingerprint": scenario.fingerprint,
        "controller": controller,
        "event_kind": scenario.event.kind,
        "event_start_time_s": scenario.event.start_time_s,
        "event_end_time_s": scenario.event.end_time_s,
        "event_value": scenario.event.value,
        **metrics,
        **windowed,
    }


def evaluate(
    *,
    scenarios: list[Scenario],
    model_path: Path,
    calibration_path: Path,
    runs_dir: Path,
    run_id: str | None = None,
) -> Path:
    validate_scenario_set(scenarios)
    model_path = model_path.resolve(strict=True)
    calibration_path = calibration_path.resolve(strict=True)
    calibration = GainCalibration.load(calibration_path)
    model_hash_before = sha256_file(model_path)
    run_id = run_id or default_run_id(scenarios)
    run_dir = create_run_directory(runs_dir, run_id)
    project_dir = Path(__file__).resolve().parent
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "controllers": list(CONTROLLERS),
        "delta_sign": "ppo_minus_fixed",
        "scenarios": [scenario.to_dict() for scenario in scenarios],
        "evaluation_config": {
            "seed_count": len({scenario.seed for scenario in scenarios}),
            "recovery_pre_window_s": config.ADAPTATION_PRE_WINDOW_S,
            "recovery_response_window_s": config.ADAPTATION_RESPONSE_WINDOW_S,
            "recovery_relative_tolerance": config.ADAPTATION_RECOVERY_RELATIVE_TOLERANCE,
            "recovery_floor_m": config.ADAPTATION_RECOVERY_FLOOR_M,
            "recovery_sustained_s": config.ADAPTATION_RECOVERY_SUSTAINED_S,
            "practical_improvement_threshold_pct": config.ADAPTATION_MEANINGFUL_IMPROVEMENT_PCT,
        },
        "artifacts": {
            "ppo_model": {"path": str(model_path), "sha256": model_hash_before},
            "pid_calibration": {
                "path": str(calibration_path),
                "sha256": sha256_file(calibration_path),
            },
        },
        "calibration": {
            "base": calibration.base.tolist(),
            "low": calibration.low.tolist(),
            "high": calibration.high.tolist(),
        },
        "source_hashes": source_hashes(project_dir),
        "versions": package_versions(),
        "expected_pairs": len(scenarios),
        "completed_pairs": 0,
        "missing_pairs": [scenario.scenario_id for scenario in scenarios],
        "errors": [],
    }
    write_json(run_dir / "manifest.json", manifest)

    path_keys = tuple(dict.fromkeys(scenario.path_key for scenario in scenarios))
    fixed_env = PathFollowingEnv(
        calibration=calibration,
        training=False,
        path_keys=path_keys,
        fixed_gains=calibration.base,
    )
    ppo_env = PathFollowingEnv(
        calibration=calibration,
        training=False,
        path_keys=path_keys,
    )
    model = PPO.load(model_path, device="cpu")
    policies = {"fixed": fixed_policy, "ppo": _policy(model)}
    environments = {"fixed": fixed_env, "ppo": ppo_env}
    episode_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    try:
        for index, scenario in enumerate(scenarios, start=1):
            pair: dict[str, dict[str, Any]] = {}
            for controller in CONTROLLERS:
                trace_path = run_dir / "traces" / f"{scenario.scenario_id}__{controller}.csv"
                try:
                    summary = run_controller(
                        environments[controller],
                        policies[controller],
                        scenario,
                        controller,
                        trace_path,
                    )
                    episode_rows.append(summary)
                    pair[controller] = summary
                except Exception as error:
                    log_path = run_dir / "logs" / f"{scenario.scenario_id}__{controller}.log"
                    log_path.write_text(traceback.format_exc(), encoding="utf-8")
                    manifest["errors"].append({
                        "scenario_id": scenario.scenario_id,
                        "controller": controller,
                        "error": f"{type(error).__name__}: {error}",
                        "log": str(log_path.relative_to(run_dir)),
                    })
            if set(pair) == set(CONTROLLERS):
                paired_rows.append(paired_row(scenario, pair["fixed"], pair["ppo"]))
            manifest["completed_pairs"] = len(paired_rows)
            completed_ids = {row["scenario_id"] for row in paired_rows}
            manifest["missing_pairs"] = [
                item.scenario_id for item in scenarios if item.scenario_id not in completed_ids
            ]
            write_csv(run_dir / "paired_summary.csv", paired_rows)
            write_json(run_dir / "manifest.json", manifest)
            print(
                f"[{index}/{len(scenarios)}] {scenario.scenario_id}: "
                f"{'paired' if scenario.scenario_id in completed_ids else 'incomplete'}"
            )
    finally:
        fixed_env.close()
        ppo_env.close()

    aggregate = aggregate_pairs(paired_rows, len(scenarios))
    aggregate["run_id"] = run_id
    aggregate["status"] = "complete" if len(paired_rows) == len(scenarios) else "incomplete"
    write_json(run_dir / "aggregate.json", aggregate)
    model_hash_after = sha256_file(model_path)
    manifest["artifacts"]["ppo_model"]["sha256_after"] = model_hash_after
    manifest["artifacts"]["ppo_model"]["unchanged"] = model_hash_before == model_hash_after
    manifest["status"] = aggregate["status"]
    manifest["completed_at"] = utc_now()
    write_json(run_dir / "manifest.json", manifest)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--matrix", choices=("smoke", "development"), default="smoke")
    parser.add_argument("--scenario-file", type=Path)
    parser.add_argument("--paths", nargs="+")
    parser.add_argument("--speeds", nargs="+", type=float)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs/adaptation_evaluation"),
    )
    return parser.parse_args()


def selected_scenarios(args: argparse.Namespace) -> list[Scenario]:
    if args.scenario_file:
        return load_scenarios(args.scenario_file)
    if args.paths or args.speeds:
        return build_scenarios(
            tuple(args.paths or DEVELOPMENT_PATHS),
            tuple(args.speeds or config.SPEED_TARGETS),
            seed=args.seed,
        )
    if args.matrix == "smoke":
        return smoke_scenarios(args.seed)
    return build_scenarios(DEVELOPMENT_PATHS, config.SPEED_TARGETS, seed=args.seed)


def main() -> None:
    args = parse_args()
    try:
        run_dir = evaluate(
            scenarios=selected_scenarios(args),
            model_path=args.model,
            calibration_path=args.calibration,
            runs_dir=args.runs_dir,
            run_id=args.run_id,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"Saved paired adaptation run to {run_dir}")


if __name__ == "__main__":
    main()
