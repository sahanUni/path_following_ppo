"""Compare observation arms against each other -- the RQ2 analysis.

`summarise_batch.py` answers RQ1: does one batch beat the fixed PID. This one
answers the different question RQ2 asks: how much does information the blind
agent does not have buy, and does it buy a scheduler or just a better constant.

Two things make the comparison cheap and fair. `confirm_advantage.py` draws its
scenarios policy-free from a fixed `--scenario-seed`, so every arm has already
replayed the identical episodes; and the fixed-gain reference involves no neural
network, so its completions must be identical across arms. That identity is
checked here before anything is compared -- if it fails, the arms were not run on
the same suite and no amount of statistics will fix it.

Both statistics from `summarise_batch` apply again, for the same reasons. Within
a seed the arms are paired on the scenario, so an exact McNemar over the
scenarios where they disagree. Across seeds a sign test over per-seed outcomes,
because episodes within a seed share a policy and pooling them would treat five
correlated runs as a thousand independent samples.

    python compare_arms.py \
        --arm blind=runs/rq1_blind \
        --arm preview=runs/rq2_preview \
        --arm context=runs/rq2_context \
        --arm both=runs/rq2_both

The first arm is the reference every other arm is tested against.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from summarise_batch import (
    baseline_name,
    exact_binomial,
    load_confirmation,
    scheduling_correlations,
)


class Arm:
    """One batch of seeds trained with one observation configuration."""

    def __init__(self, name: str, root: Path):
        self.name = name
        self.root = root
        self.seeds: dict[int, dict[str, Any]] = {}

    def load(self, filename: str) -> None:
        files = sorted(self.root.glob(f"seed*/{filename}"))
        if not files:
            raise SystemExit(f"no {filename} under {self.root}/seed*/ for arm '{self.name}'")
        for path in files:
            digits = "".join(c for c in path.parent.name if c.isdigit())
            if not digits:
                raise SystemExit(f"cannot read a seed number from {path.parent}")
            data, scenarios = load_confirmation(path)
            if "PPO" not in data:
                raise SystemExit(f"{path} has no PPO rows")
            fixed = data[baseline_name(list(data))]
            self.seeds[int(digits)] = {
                "ppo": data["PPO"],
                "fixed": fixed,
                "scenarios": scenarios,
                "path": path,
            }


def parse_arm(token: str) -> Arm:
    """`name=path`, or a bare path whose directory name becomes the label."""
    if "=" in token:
        name, _, raw = token.partition("=")
        return Arm(name.strip(), Path(raw.strip()))
    root = Path(token)
    return Arm(root.name, root)


def completion_vector(entry: dict[str, Any], key: str, scenarios: list[int]) -> tuple[int, ...]:
    return tuple(int(entry[key][s]["finished"]) for s in scenarios)


def check_the_arms_ran_the_same_suite(arms: list[Arm]) -> str:
    """The fixed-gain rows are the control: no network, so no excuse to differ.

    A mismatch means the arms were run against different scenario seeds, path
    sets, or calibrations. Comparing them then measures the suite, not the
    observation, and the arm with the kinder episodes wins for the wrong reason.
    """
    signatures: dict[tuple[int, ...], list[str]] = {}
    for arm in arms:
        for seed, entry in arm.seeds.items():
            order = sorted(entry["ppo"])
            signatures.setdefault(
                completion_vector(entry, "fixed", order), []
            ).append(f"{arm.name}/seed{seed}")
    if len(signatures) > 1:
        lines = [
            "the fixed-gain baseline does not agree across arms, so they did "
            "not run the same scenario suite:"
        ]
        for vector, owners in signatures.items():
            lines.append(f"  {sum(vector)} completions: {', '.join(sorted(owners))}")
        lines.append(
            "Re-run confirm_advantage.py with the same --scenario-seed, "
            "--episodes and --calibration for every arm."
        )
        raise SystemExit("\n".join(lines))
    vector = next(iter(signatures))
    return f"{sum(vector)}/{len(vector)} on every arm and seed"


def common_finished(arms: list[Arm], seed: int) -> list[int]:
    """Scenarios every arm's policy finished on this seed.

    Tracking has to be read on a shared episode set. Averaging each arm over its
    own completions compares different episodes, and the arm that fails the hard
    ones looks tidier for having failed them.
    """
    order = sorted(arms[0].seeds[seed]["ppo"])
    return [
        s for s in order
        if all(arm.seeds[seed]["ppo"][s]["finished"] for arm in arms)
    ]


def per_arm_table(arms: list[Arm], seeds: list[int]) -> None:
    print("per arm, across seeds (cm on the episodes EVERY arm finished)")
    print(f"{'arm':>10}{'seeds':>7}{'PPO completion':>22}{'fixed':>9}"
          f"{'gain':>8}{'cm':>8}{'mean Kp':>9}")
    for arm in arms:
        rates, fixed_rates, cms, kps = [], [], [], []
        for seed in seeds:
            entry = arm.seeds[seed]
            order = sorted(entry["ppo"])
            shared = common_finished(arms, seed)
            rates.append(sum(entry["ppo"][s]["finished"] for s in order) / len(order))
            fixed_rates.append(sum(entry["fixed"][s]["finished"] for s in order) / len(order))
            cms.append(np.mean([entry["ppo"][s]["distance"] for s in shared]) * 100
                       if shared else np.nan)
            kps.append(np.mean([entry["ppo"][s]["mean_kp"] for s in order]))
        rate = np.array(rates)
        spread = rate.std(ddof=1) if len(rate) > 1 else 0.0
        print(f"{arm.name:>10}{len(seeds):>7}"
              f"{rate.mean():>15.3f} +/-{spread:>6.3f}"
              f"{np.mean(fixed_rates):>9.3f}{rate.mean() - np.mean(fixed_rates):>+8.3f}"
              f"{np.nanmean(cms):>8.2f}{np.mean(kps):>9.1f}")


def paired_table(arms: list[Arm], seeds: list[int]) -> None:
    reference = arms[0]
    for arm in arms[1:]:
        print(f"\n{arm.name} against {reference.name}, paired on the scenario")
        print(f"{'seed':>8}{arm.name:>12}{reference.name:>12}{'win':>6}{'loss':>6}"
              f"{'p':>10}{'cm':>8}{'cm ref':>9}")
        wins_by_seed = 0
        losses_by_seed = 0
        for seed in seeds:
            mine = arm.seeds[seed]["ppo"]
            theirs = reference.seeds[seed]["ppo"]
            order = sorted(mine)
            wins = sum(1 for s in order if mine[s]["finished"] and not theirs[s]["finished"])
            losses = sum(1 for s in order if theirs[s]["finished"] and not mine[s]["finished"])
            mine_done = sum(mine[s]["finished"] for s in order)
            theirs_done = sum(theirs[s]["finished"] for s in order)
            both = [s for s in order if mine[s]["finished"] and theirs[s]["finished"]]
            mine_cm = np.mean([mine[s]["distance"] for s in both]) * 100 if both else np.nan
            theirs_cm = np.mean([theirs[s]["distance"] for s in both]) * 100 if both else np.nan
            wins_by_seed += mine_done > theirs_done
            losses_by_seed += mine_done < theirs_done
            print(f"{seed:>8}{mine_done:>8}/{len(order):<3}{theirs_done:>8}/{len(order):<3}"
                  f"{wins:>6}{losses:>6}{exact_binomial(wins, losses):>10.4f}"
                  f"{mine_cm:>8.2f}{theirs_cm:>9.2f}")
        sign = exact_binomial(wins_by_seed, losses_by_seed)
        print(f"  seeds where {arm.name} wins: {wins_by_seed}/{len(seeds)}"
              + (f", loses: {losses_by_seed}" if losses_by_seed else "")
              + f"   sign test p = {sign:.4f}")
        if sign >= 0.05:
            print(f"  not separable from {reference.name} at this seed count -- "
                  "report the per-seed table, not the mean")


def scheduling_table(arms: list[Arm], seeds: list[int]) -> None:
    """The RQ2 headline, and the reason the privileged arm exists at all.

    A policy that has identified the plant lowers its gain when dead time is
    high: a clearly NEGATIVE correlation. The blind agent measures about +0.08
    with a narrow gain spread, i.e. it found a constant and reacts within the
    episode. If the privileged arm does not move that number, the information
    reached the observation but never reached the behaviour, and a completion
    win it happens to show is not evidence of scheduling.
    """
    print("\nis the arm scheduling, or picking a constant?"
          "  corr(mean episode Kp, hidden parameter), averaged over seeds")
    print(f"{'arm':>10}{'delay':>9}{'mass':>9}{'friction':>10}{'noise':>8}"
          f"{'Kp spread':>12}{'worst seed':>12}")
    for arm in arms:
        columns: dict[str, list[float]] = {k: [] for k in ("delay", "mass", "friction", "noise")}
        spreads, delays = [], []
        for seed in seeds:
            entry = arm.seeds[seed]
            correlations = scheduling_correlations(entry["ppo"], entry["scenarios"])
            for key in columns:
                columns[key].append(correlations[key])
            spreads.append(correlations["gain_spread"])
            delays.append(correlations["delay"])
        weakest = max(delays)  # least negative, i.e. least evidence of scheduling
        print(f"{arm.name:>10}{np.nanmean(columns['delay']):>+9.2f}"
              f"{np.nanmean(columns['mass']):>+9.2f}"
              f"{np.nanmean(columns['friction']):>+10.2f}"
              f"{np.nanmean(columns['noise']):>+8.2f}"
              f"{np.mean(spreads):>12.1f}{weakest:>+12.2f}")
    print("  scheduling shows up as a clearly negative delay column in EVERY")
    print("  seed; near zero means the arm settled on a constant whatever the")
    print("  plant was, and any advantage comes from where that constant sits.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--arm", action="append", required=True, metavar="NAME=RUN_ROOT",
        help=("Repeatable. The FIRST arm is the reference the others are tested "
              "against -- normally the blind one."),
    )
    parser.add_argument("--filename", type=str, default="confirm_200.csv")
    parser.add_argument(
        "--allow-partial-seeds", action="store_true",
        help=("Compare the seeds the arms share instead of refusing. Convenient "
              "mid-batch; do not report a table built this way without saying so."),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arms = [parse_arm(token) for token in args.arm]
    if len(arms) < 2:
        raise SystemExit("give at least two --arm entries; one arm compares to nothing")
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names):
        raise SystemExit(f"arm names must be distinct: {names}")
    for arm in arms:
        arm.load(args.filename)

    shared_seeds = set(arms[0].seeds)
    for arm in arms[1:]:
        shared_seeds &= set(arm.seeds)
    if not shared_seeds:
        raise SystemExit("the arms have no seed in common: "
                         + ", ".join(f"{a.name}={sorted(a.seeds)}" for a in arms))
    missing = {arm.name: sorted(set(arm.seeds) - shared_seeds) for arm in arms}
    if any(missing.values()):
        report = ", ".join(f"{name} has extra {seeds}" for name, seeds in missing.items() if seeds)
        if not args.allow_partial_seeds:
            raise SystemExit(
                f"the arms do not cover the same seeds ({report}). Finish the "
                "batch, or pass --allow-partial-seeds to compare the "
                f"{len(shared_seeds)} they share."
            )
        print(f"WARNING: comparing only the shared seeds; {report}\n")
    seeds = sorted(shared_seeds)

    episodes = len(arms[0].seeds[seeds[0]]["ppo"])
    for arm in arms:
        for seed in seeds:
            if len(arm.seeds[seed]["ppo"]) != episodes:
                raise SystemExit(
                    f"{arm.seeds[seed]['path']} has {len(arm.seeds[seed]['ppo'])} "
                    f"scenarios, expected {episodes}; the arms are not paired"
                )

    print(f"{len(arms)} arms x {len(seeds)} seeds x {episodes} paired scenarios")
    for arm in arms:
        print(f"  {arm.name:>10}  {arm.root}  seeds {seeds}")
    print(f"\nintegrity: fixed-gain baseline completes "
          f"{check_the_arms_ran_the_same_suite(arms)}\n")

    per_arm_table(arms, seeds)
    paired_table(arms, seeds)
    scheduling_table(arms, seeds)


if __name__ == "__main__":
    main()
