"""Aggregate a multi-seed batch into the numbers that go in the write-up.

A single seed answers "is this model better than the fixed baseline". A batch
answers the different and more important question: "does training RELIABLY
produce such a model, or did one seed get lucky". Those need different
statistics, so both are reported here and kept apart.

Per seed, the paired exact McNemar test over the scenarios where the two
controllers disagree -- the ones both finish, or both fail, carry no
information about which is better.

Across seeds, a sign test over per-seed outcomes. It deliberately ignores the
size of each seed's win: pooling all episodes from all seeds would treat five
seeds x 200 episodes as 1000 independent samples, which overstates the evidence
badly, because episodes within a seed share a policy and are not independent of
each other in the way the test would assume.

    python summarise_batch.py --run-root runs/rq1_blind
"""

from __future__ import annotations

import argparse
import csv
from math import comb
from pathlib import Path
from typing import Any

import numpy as np


def exact_binomial(wins: int, losses: int) -> float:
    """Two-sided probability of a split this lopsided under a fair coin."""
    total = wins + losses
    if total == 0:
        return 1.0
    extreme = min(wins, losses)
    tail = sum(comb(total, k) for k in range(extreme + 1)) / (2.0**total)
    return min(1.0, 2.0 * tail)


def load_confirmation(path: Path) -> tuple[
    dict[str, dict[int, dict[str, Any]]], dict[int, dict[str, float]]
]:
    by_controller: dict[str, dict[int, dict[str, Any]]] = {}
    scenarios: dict[int, dict[str, float]] = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        scenario = int(row["scenario"])
        by_controller.setdefault(row["controller"], {})[scenario] = {
            "finished": int(row["finished"]),
            "distance": float(row["mean_distance_m"]),
            "mean_kp": float(row["mean_kp"]),
        }
        scenarios[scenario] = {
            "delay": float(row["actuator_delay_s"]),
            "mass": float(row["mass"]),
            "friction": float(row["friction"]),
            "noise": float(row["sensor_noise_m"]),
        }
    return by_controller, scenarios


def scheduling_correlations(
    ppo: dict[int, dict[str, Any]], scenarios: dict[int, dict[str, float]]
) -> dict[str, float]:
    """Does the episode's chosen gain track the episode's hidden parameters?

    This is what separates a scheduler from a policy that found a good constant.
    A blind agent that has genuinely inferred the plant should lower its gain
    when dead time is high -- a clearly NEGATIVE correlation. Near zero means it
    is picking much the same gain whatever the plant is, and any advantage it
    shows comes from where that constant sits rather than from adapting.
    """
    keys = sorted(ppo)
    gains = np.array([ppo[k]["mean_kp"] for k in keys])
    out: dict[str, float] = {}
    for name in ("delay", "mass", "friction", "noise"):
        values = np.array([scenarios[k][name] for k in keys])
        if gains.std() < 1e-9 or values.std() < 1e-9:
            out[name] = float("nan")
        else:
            out[name] = float(np.corrcoef(gains, values)[0, 1])
    out["gain_spread"] = float(gains.max() - gains.min())
    return out


def baseline_name(controllers: list[str]) -> str:
    for name in controllers:
        if "baseline" in name:
            return name
    fixed = [n for n in controllers if n != "PPO"]
    if not fixed:
        raise SystemExit("no fixed-gain controller found to compare against")
    return fixed[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--filename", type=str, default="confirm_200.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)
    files = sorted(
        root.glob(f"seed*/{args.filename}"),
        key=lambda p: int("".join(c for c in p.parent.name if c.isdigit()) or 0),
    )
    if not files:
        raise SystemExit(f"no {args.filename} under {root}/seed*/")

    print(f"{len(files)} seeds from {root}\n")
    per_seed: list[dict[str, Any]] = []
    controllers: list[str] = []

    for path in files:
        seed = path.parent.name
        data, scenario_meta = load_confirmation(path)
        controllers = list(data)
        base = baseline_name(controllers)
        ppo, fixed = data["PPO"], data[base]
        scenarios = sorted(ppo)

        wins = sum(1 for s in scenarios if ppo[s]["finished"] and not fixed[s]["finished"])
        losses = sum(1 for s in scenarios if fixed[s]["finished"] and not ppo[s]["finished"])
        ppo_done = sum(ppo[s]["finished"] for s in scenarios)
        fixed_done = sum(fixed[s]["finished"] for s in scenarios)

        # Tracking is compared only on scenarios BOTH finish. Averaging over
        # each controller's own completions compares different episode sets:
        # the controller that fails the hard ones looks tidier for it.
        shared = [s for s in scenarios if ppo[s]["finished"] and fixed[s]["finished"]]
        ppo_cm = np.mean([ppo[s]["distance"] for s in shared]) * 100 if shared else np.nan
        fixed_cm = np.mean([fixed[s]["distance"] for s in shared]) * 100 if shared else np.nan

        per_seed.append({
            "seed": seed,
            "n": len(scenarios),
            "ppo_done": ppo_done,
            "fixed_done": fixed_done,
            "wins": wins,
            "losses": losses,
            "p": exact_binomial(wins, losses),
            "ppo_cm": ppo_cm,
            "fixed_cm": fixed_cm,
            "mean_kp": np.mean([ppo[s]["mean_kp"] for s in scenarios]),
            "corr": scheduling_correlations(ppo, scenario_meta),
        })

    base = baseline_name(controllers)
    print(f"PPO versus {base}, per seed")
    print(f"{'seed':>8}{'PPO':>10}{'fixed':>10}{'win':>6}{'loss':>6}"
          f"{'p':>10}{'PPO cm':>9}{'fixed cm':>10}{'mean Kp':>9}")
    for row in per_seed:
        print(f"{row['seed']:>8}{row['ppo_done']:>6}/{row['n']:<3}"
              f"{row['fixed_done']:>6}/{row['n']:<3}{row['wins']:>6}{row['losses']:>6}"
              f"{row['p']:>10.4f}{row['ppo_cm']:>9.2f}{row['fixed_cm']:>10.2f}"
              f"{row['mean_kp']:>9.1f}")

    rates = np.array([r["ppo_done"] / r["n"] for r in per_seed])
    fixed_rates = np.array([r["fixed_done"] / r["n"] for r in per_seed])
    better = sum(1 for r in per_seed if r["ppo_done"] > r["fixed_done"])
    worse = sum(1 for r in per_seed if r["ppo_done"] < r["fixed_done"])

    print(f"\nacross {len(per_seed)} seeds")
    print(f"  PPO completion   : {rates.mean():.3f} +/- {rates.std(ddof=1):.3f} "
          f"(min {rates.min():.3f}, max {rates.max():.3f})")
    print(f"  fixed completion : {fixed_rates.mean():.3f} "
          f"+/- {fixed_rates.std(ddof=1):.3f}")
    print(f"  seeds where PPO wins: {better}/{len(per_seed)}"
          + (f", loses: {worse}" if worse else ""))
    print(f"  sign test across seeds: p = {exact_binomial(better, worse):.4f}")

    significant = sum(1 for r in per_seed if r["p"] < 0.05)
    print(f"  seeds individually significant at 0.05: {significant}/{len(per_seed)}")

    # Scheduling, not just performance. A policy that has inferred the plant
    # lowers its gain when dead time is high; near-zero correlation with a
    # narrow gain spread means it settled on a constant, and any advantage
    # comes from where that constant landed rather than from adapting.
    print(f"\nis it scheduling, or just picking a constant?")
    print(f"{'seed':>8}{'corr Kp/delay':>15}{'corr Kp/mass':>14}"
          f"{'Kp spread':>12}{'mean Kp':>9}")
    for row in per_seed:
        c = row["corr"]
        print(f"{row['seed']:>8}{c['delay']:>15.2f}{c['mass']:>14.2f}"
              f"{c['gain_spread']:>12.1f}{row['mean_kp']:>9.1f}")
    print("  a scheduler should be clearly NEGATIVE on delay;"
          " near zero means a constant")

    spread = rates.max() - rates.min()
    print(f"\n  seed-to-seed spread in PPO completion: {spread:.3f}")
    if spread > (rates.mean() - fixed_rates.mean()):
        print("  NOTE: the spread across seeds exceeds the mean advantage over the")
        print("  baseline. The effect is real only if it survives that variance --")
        print("  report the per-seed table, not just the mean.")


if __name__ == "__main__":
    main()
