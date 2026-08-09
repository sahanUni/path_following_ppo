"""Search for one robust fixed steering PID and derive PPO gain bounds."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np

from calibration import GainCalibration
import config
from env import PathFollowingEnv


@dataclass(frozen=True)
class Scenario:
    path_key: str
    v_target: float
    mass: float
    friction: float
    actuator: float
    actuator_delay_s: float
    sensor_noise_m: float
    noise_seed: int


def make_candidates(count: int, seed: int) -> list[np.ndarray]:
    """Deterministic mixture of known sensible points and broad random search."""
    if count < 1:
        raise ValueError("candidate count must be positive")
    anchors = [
        (1.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (8.0, 0.0, 0.0),
        (15.0, 0.0, 0.0),
        (25.0, 0.0, 0.0),
        (8.0, 0.0, 0.3),
        (15.0, 0.25, 0.5),
        (25.0, 0.5, 1.0),
    ]
    candidates = [np.asarray(x, dtype=np.float64) for x in anchors[:count]]
    rng = np.random.default_rng(seed)
    while len(candidates) < count:
        kp = float(np.exp(rng.uniform(np.log(0.25), np.log(50.0))))
        ki = 0.0 if rng.random() < 0.35 else float(rng.uniform(0.0, 5.0))
        kd = 0.0 if rng.random() < 0.25 else float(np.exp(rng.uniform(np.log(0.01), np.log(10.0))))
        candidates.append(np.array([kp, ki, kd], dtype=np.float64))
    return candidates


def make_scenarios(
    path_keys: list[str],
    speeds: list[float],
    physics_samples: int,
    seed: int,
) -> list[Scenario]:
    # The nominal sample keeps a clean, delay-free reference point in the mix.
    # The randomized ones carry dead time and sensor noise, WITHOUT which the
    # whole sweep runs on the plant where raising Kp is free: the winning
    # candidate would then be whichever gain saturates the wheels first, and the
    # action box derived from it would be centred on the wrong answer.
    physics = [
        (
            config.NOMINAL_MASS_KG,
            config.NOMINAL_FRICTION,
            config.NOMINAL_ACTUATOR,
            0.0,
            0.0,
        )
    ]
    rng = np.random.default_rng(seed)
    for _ in range(physics_samples):
        physics.append(
            (
                float(rng.uniform(*config.MASS_RANGE_KG)),
                float(rng.uniform(*config.FRICTION_RANGE)),
                float(rng.uniform(*config.ACTUATOR_RANGE)),
                float(rng.uniform(*config.ACTUATOR_DELAY_RANGE_S)),
                float(rng.uniform(*config.SENSOR_NOISE_RANGE_M)),
            )
        )
    return [
        Scenario(
            path_key,
            float(speed),
            mass,
            friction,
            actuator,
            delay_s,
            noise_m,
            # Fixed per scenario, so every candidate gain meets the identical
            # sensor trace. A per-candidate draw would score gains against
            # different noise and rank partly on luck.
            int(seed),
        )
        for path_key in path_keys
        for speed in speeds
        for mass, friction, actuator, delay_s, noise_m in physics
    ]


def run_fixed_episode(gains: np.ndarray, scenario: Scenario) -> dict:
    env = PathFollowingEnv(
        training=False,
        path_keys=(scenario.path_key,),
        fixed_gains=gains,
    )
    _, _ = env.reset(
        seed=config.SEED,
        options={
            "path_key": scenario.path_key,
            "v_target": scenario.v_target,
            "mass": scenario.mass,
            "friction": scenario.friction,
            "actuator": scenario.actuator,
            "actuator_delay_s": scenario.actuator_delay_s,
            "sensor_noise_m": scenario.sensor_noise_m,
            "noise_seed": scenario.noise_seed,
            "stage": 2,
        },
    )
    terminated = truncated = False
    info: dict = {}
    zero_action = np.zeros(3, dtype=np.float32)
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(zero_action)
    env.close()
    return dict(info["episode_metrics"])


def summarize_candidate(gains: np.ndarray, episodes: list[dict]) -> dict:
    completed = sum(int(row["finished"]) for row in episodes)
    return {
        "kp": float(gains[0]),
        "ki": float(gains[1]),
        "kd": float(gains[2]),
        "completed": completed,
        "completion_rate": completed / len(episodes),
        "mean_distance_m": float(np.mean([x["mean_distance_m"] for x in episodes])),
        "mean_itae": float(np.mean([x["itae"] for x in episodes])),
        "mean_saturation_fraction": float(
            np.mean([x["saturation_fraction"] for x in episodes])
        ),
    }


def rank_key(row: dict) -> tuple[float, float, float, float]:
    """Completion dominates distance, then ITAE, then smaller gain magnitude.

    Kept for the single-row ordering it describes, but `rank_rows` is what the
    calibration actually uses: a strict ordering on completion lets a gain that
    merely stays inside the corridor outrank one that tracks several times
    better. See `config.COMPLETION_TOLERANCE`.
    """
    gain_norm = row["kp"] + row["ki"] + row["kd"]
    return (-row["completed"], row["mean_distance_m"], row["mean_itae"], gain_norm)


def completion_threshold(rows: list[dict]) -> float:
    """Completion count at or above which candidates are treated as tied."""
    if not rows:
        return 0.0
    return config.COMPLETION_TOLERANCE * max(row["completed"] for row in rows)


def rank_rows(rows: list[dict]) -> list[dict]:
    """Best first: robust-enough candidates ordered by how well they track.

    Two tiers. Everything within `config.COMPLETION_TOLERANCE` of the best
    completion count is tier 0 and is ranked purely on tracking; everything
    else is tier 1 and can never outrank tier 0 however tightly it tracks. That
    keeps completion decisive across the big gap -- a gain that fails a third of
    the suite is still disqualified -- without letting a single unlucky physics
    sample pick the baseline.
    """
    threshold = completion_threshold(rows)

    def key(row: dict) -> tuple[float, ...]:
        return (
            0 if row["completed"] >= threshold else 1,
            row["mean_distance_m"],
            row["mean_itae"],
            row["kp"] + row["ki"] + row["kd"],
        )

    return sorted(rows, key=key)


def derive_calibration(
    rows: list[dict],
    metadata: dict,
    box_low: np.ndarray | None = None,
    box_high: np.ndarray | None = None,
) -> GainCalibration:
    """Pick the baseline gains, and either derive or accept the action box.

    `box_low`/`box_high` set the box explicitly. That is the honest option when
    the candidate sweep has told you where the optimum MOVES but has not sampled
    a wide enough spread of survivors to bound it: the box's job is coverage of
    that range, not optimality, and padding a single winning candidate by a
    fixed margin only looks derived. Whichever route, the two-sided check runs.
    """
    if not rows:
        raise ValueError("cannot calibrate without candidate results")
    ranked = rank_rows(rows)
    base = np.array([ranked[0]["kp"], ranked[0]["ki"], ranked[0]["kd"]])

    if box_low is not None or box_high is not None:
        if box_low is None or box_high is None:
            raise ValueError("an explicit gain box needs both low and high")
        low = np.minimum(np.asarray(box_low, dtype=np.float64), base)
        high = np.maximum(np.asarray(box_high, dtype=np.float64), base)
        if np.any(low < config.CALIBRATION_SEARCH_LOW) or np.any(
            high > config.CALIBRATION_SEARCH_HIGH
        ):
            raise ValueError("the explicit gain box falls outside the search space")
        calibration = GainCalibration(
            base=base,
            low=low,
            high=high,
            metadata={**metadata, "box_source": "explicit"},
        )
        assert_two_sided(calibration)
        return calibration

    # Everything tied on completion is a candidate for bounding the box, not
    # only the rows matching the winner's exact count -- with a hard plant the
    # top count is often reached by a single candidate, and the box then
    # collapses onto the baseline and is rebuilt entirely out of MIN_GAIN_MARGIN.
    threshold = completion_threshold(rows)
    same_completion = [row for row in ranked if row["completed"] >= threshold]
    useful_count = max(1, int(math.ceil(len(same_completion) * 0.25)))
    useful = same_completion[:useful_count]
    gains = np.array([[x["kp"], x["ki"], x["kd"]] for x in useful])
    low = np.minimum(gains.min(axis=0), base)
    high = np.maximum(gains.max(axis=0), base)
    missing_span = np.maximum(0.0, config.MIN_GAIN_SPAN - (high - low))
    low = np.maximum(config.CALIBRATION_SEARCH_LOW, low - 0.5 * missing_span)
    high = np.minimum(config.CALIBRATION_SEARCH_HIGH, high + 0.5 * missing_span)
    # If clipping removed span near zero, extend the opposite side.
    high = np.maximum(high, np.minimum(config.CALIBRATION_SEARCH_HIGH, low + config.MIN_GAIN_SPAN))
    low = np.minimum(low, np.maximum(config.CALIBRATION_SEARCH_LOW, high - config.MIN_GAIN_SPAN))
    low = np.minimum(low, base)
    high = np.maximum(high, base)

    # Both sides of the baseline must stay reachable. Without this the winning
    # candidate being the most extreme survivor on an axis silently welds that
    # axis shut: gains_from_action maps the whole positive half of the channel
    # to `base + action * (high - base)`, which is identically zero when
    # high == base. The policy keeps emitting actions, none of them do anything,
    # and nothing anywhere reports a problem.
    margin = np.maximum(config.MIN_GAIN_MARGIN * np.abs(base), 0.5 * config.MIN_GAIN_SPAN)
    high = np.minimum(config.CALIBRATION_SEARCH_HIGH, np.maximum(high, base + margin))
    low = np.maximum(config.CALIBRATION_SEARCH_LOW, np.minimum(low, base - margin))

    calibration = GainCalibration(
        base=base, low=low, high=high, metadata={**metadata, "box_source": "derived"}
    )
    assert_two_sided(calibration)
    return calibration


def assert_two_sided(calibration: GainCalibration) -> None:
    """Fail loudly if any action channel cannot move its gain in some direction.

    A side that is welded shut *at the edge of the search space* is a physical
    constraint, not a derivation fault -- a baseline Ki of 0 has nowhere lower
    to go. That case is reported and allowed. A side welded shut anywhere in
    the interior is the bug this function exists to catch.
    """
    for index, name in enumerate(("Kp", "Ki", "Kd")):
        base = calibration.base[index]
        for bound, limit, verb, past in (
            (calibration.high[index], config.CALIBRATION_SEARCH_HIGH[index], "raise", "raised"),
            (calibration.low[index], config.CALIBRATION_SEARCH_LOW[index], "lower", "lowered"),
        ):
            if (bound > base) if verb == "raise" else (bound < base):
                continue
            if math.isclose(bound, limit, rel_tol=1e-9, abs_tol=1e-9):
                print(
                    f"  note: {name} cannot be {past} -- baseline {base:.4f} "
                    f"already sits at the search limit {limit:.4f}"
                )
                continue
            raise ValueError(
                f"{name}: the policy can never {verb} this gain -- bound "
                f"{bound:.4f} does not clear baseline {base:.4f}, and the search "
                f"limit {limit:.4f} left room. The action channel maps to nothing."
            )


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _floats(text: str) -> list[float]:
    values = [float(part) for part in str(text).replace(",", " ").split()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one number")
    return values


def _triple(text: str) -> np.ndarray:
    values = _floats(text)
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected exactly three numbers: Kp,Ki,Kd")
    return np.asarray(values, dtype=np.float64)


def grid_candidates(
    kp_values: list[float], ki_values: list[float], kd_values: list[float]
) -> list[np.ndarray]:
    """Every combination of the requested gains, in a stable order.

    A targeted grid instead of the log-uniform random search, for when earlier
    evidence has already located the region worth searching. The v6 sweep drew
    48 random candidates and put exactly ONE of them in the region that turned
    out to matter (Kp above 25 with Ki near zero), so its winner was a property
    of the sampling rather than of the plant.
    """
    return [
        np.array([kp, ki, kd], dtype=np.float64)
        for kp in kp_values
        for ki in ki_values
        for kd in kd_values
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="+", default=list(config.TRAIN_PATHS))
    parser.add_argument("--speeds", nargs="+", type=float, default=list(config.SPEED_TARGETS))
    parser.add_argument("--candidates", type=int, default=48)
    parser.add_argument(
        "--kp-grid",
        type=_floats,
        default=None,
        help=(
            "Comma-separated Kp values. Giving any of --kp-grid/--ki-grid/"
            "--kd-grid replaces the random search with their cross product."
        ),
    )
    parser.add_argument("--ki-grid", type=_floats, default=None)
    parser.add_argument("--kd-grid", type=_floats, default=None)
    parser.add_argument(
        "--box-low",
        type=_triple,
        default=None,
        help=(
            "Set the action box explicitly as Kp,Ki,Kd instead of deriving it "
            "from the surviving candidates. Use when the sweep has located the "
            "range the optimum MOVES over but has too few survivors to bound "
            "it. Requires --box-high."
        ),
    )
    parser.add_argument("--box-high", type=_triple, default=None)
    parser.add_argument("--physics-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--output", type=Path, default=Path("runs/calibration.json"))
    parser.add_argument(
        "--from-candidates",
        type=Path,
        default=None,
        help=(
            "Re-derive the gain box from a saved *_candidates.csv instead of "
            "re-running the sweep. The candidate scores are what cost the "
            "compute; the box is just arithmetic over them."
        ),
    )
    return parser.parse_args()


NUMERIC_CANDIDATE_FIELDS = (
    "kp",
    "ki",
    "kd",
    "completed",
    "completion_rate",
    "mean_distance_m",
    "mean_itae",
    "mean_saturation_fraction",
)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no candidates")
    for row in rows:
        for field in NUMERIC_CANDIDATE_FIELDS:
            row[field] = float(row[field])
        row["completed"] = int(row["completed"])
    return rows


def main() -> None:
    args = parse_args()
    if args.from_candidates is not None:
        rows = read_rows(args.from_candidates)
        print(f"Re-deriving the gain box from {len(rows)} saved candidates (no episodes run)")
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "derived_from": str(args.from_candidates),
            "candidate_count": len(rows),
            "selection": "completion, mean distance, ITAE, gain magnitude",
        }
    else:
        grids = (args.kp_grid, args.ki_grid, args.kd_grid)
        if any(grid is not None for grid in grids):
            kp_values, ki_values, kd_values = (
                grid if grid is not None else [0.0] for grid in grids
            )
            candidates = grid_candidates(list(kp_values), list(ki_values), list(kd_values))
        else:
            candidates = make_candidates(args.candidates, args.seed)
        scenarios = make_scenarios(args.paths, args.speeds, args.physics_samples, args.seed)
        print(f"Calibrating {len(candidates)} candidates on {len(scenarios)} scenarios")
        rows = []
        for index, gains in enumerate(candidates, start=1):
            episodes = [run_fixed_episode(gains, scenario) for scenario in scenarios]
            row = summarize_candidate(gains, episodes)
            rows.append(row)
            print(
                f"[{index:>3}/{len(candidates)}] gains={gains.round(3).tolist()} "
                f"complete={row['completed']}/{len(scenarios)} "
                f"mean_dist={row['mean_distance_m']:.4f} m"
            )
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "paths": args.paths,
            "speeds": args.speeds,
            "physics_samples": args.physics_samples,
            "candidate_count": len(candidates),
            "scenario_count": len(scenarios),
            "selection": (
                f"completion within {config.COMPLETION_TOLERANCE:.0%} of best, "
                "then mean distance, ITAE, gain magnitude"
            ),
        }
    # Written BEFORE the box is derived, and deliberately so. The candidate
    # scores are the hours; the box is arithmetic over them. Deriving first
    # meant a rejected box -- a mistyped --box-high that welds a channel, say --
    # threw the whole sweep away with nothing on disk. Now the sweep survives
    # any box error and can be re-derived for free with --from-candidates.
    candidates_csv = args.output.with_name(args.output.stem + "_candidates.csv")
    if args.from_candidates is None:
        write_rows(candidates_csv, rank_rows(rows))
        print(f"Saved {candidates_csv}")

    calibration = derive_calibration(rows, metadata, args.box_low, args.box_high)
    calibration.save(args.output)
    print(f"Saved {args.output}")
    print(f"box source: {calibration.metadata.get('box_source')}")
    print(f"baseline: {calibration.base.round(4).tolist()}")
    print(f"bounds:   {calibration.low.round(4).tolist()} -> {calibration.high.round(4).tolist()}")


if __name__ == "__main__":
    main()
