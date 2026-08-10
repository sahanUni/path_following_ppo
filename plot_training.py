"""Training curves, and a plain answer to "was that enough training?".

Two curves, and they say different things:

* The EVALUATION curve is the one that answers the question. It runs a fixed
  deterministic suite at every checkpoint, so the task never changes and a
  flat tail really does mean the model stopped improving.

* The TRAINING curve does NOT answer it, and read naively it is misleading.
  The curriculum makes episodes harder as training proceeds -- nominal physics
  until 20% of the budget, randomised physics to 40%, disturbances after that
  -- so tracking error RISES with timesteps even while the policy improves. The
  curriculum boundaries are shaded here for exactly that reason.

The verdict at the end compares the last quarter of evaluations against the
quarter before it. If the change is smaller than the spread between seeds, more
of the same training would not have helped, and that is the defensible answer.

    python plot_training.py --run-root runs/rq1_blind --run-root runs/rq2_context
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import config


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_monitor(path: Path) -> list[dict[str, str]]:
    """monitor.csv carries a JSON comment line before the header."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()
    return list(csv.DictReader(lines[1:]))


def column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            values.append(np.nan)
    return np.array(values, dtype=np.float64)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average, so a trend is visible through episode noise."""
    if len(values) < window or window < 2:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_seed(directory: Path) -> dict[str, Any] | None:
    evaluation = read_csv(directory / "evaluation_history.csv")
    episodes = read_csv(directory / "episodes.csv")
    monitor = read_monitor(directory / "monitor.csv")
    if not evaluation and not episodes:
        return None
    return {
        "name": directory.name,
        "evaluation": evaluation,
        "episodes": episodes,
        "monitor": monitor,
    }


def convergence(seeds: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    """Change over the final quarter of training, against the seed spread.

    Plateau is not "the curve looks flat" -- it is that the remaining change is
    smaller than the variation between seeds, i.e. indistinguishable from which
    run you happened to look at.
    """
    late, earlier, finals = [], [], []
    for seed in seeds:
        values = column(seed["evaluation"], key)
        values = values[np.isfinite(values)]
        if len(values) < 8:
            continue
        quarter = max(len(values) // 4, 2)
        late.append(float(np.mean(values[-quarter:])))
        earlier.append(float(np.mean(values[-2 * quarter : -quarter])))
        finals.append(float(values[-1]))
    if not late:
        return None
    change = np.array(late) - np.array(earlier)
    return {
        "change": float(np.mean(change)),
        "change_spread": float(np.std(change, ddof=1)) if len(change) > 1 else 0.0,
        "seed_spread": float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
        "final": float(np.mean(finals)),
        "seeds": len(late),
    }


def curriculum_bounds(seeds: list[dict[str, Any]], run_root: Path) -> tuple[float, float]:
    manifest = run_root / "batch_manifest.json"
    total = None
    if manifest.exists():
        total = json.loads(manifest.read_text(encoding="utf-8")).get("total_timesteps")
    if not total:
        steps = [column(s["evaluation"], "timesteps") for s in seeds if s["evaluation"]]
        total = max((float(np.nanmax(v)) for v in steps if len(v)), default=1.0)
    return (
        total * config.CURRICULUM_NOMINAL_END,
        total * config.CURRICULUM_RANDOMIZED_END,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/training_curves.png"))
    parser.add_argument("--smooth", type=int, default=25,
                        help="Episodes per point on the training curves.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms: list[tuple[str, Path, list[dict[str, Any]]]] = []
    for root in args.run_root:
        seeds = [s for s in (load_seed(d) for d in sorted(root.glob("seed*"))) if s]
        if not seeds:
            print(f"  no seed data under {root}, skipping")
            continue
        arms.append((root.name, root, seeds))
    if not arms:
        raise SystemExit("no run data found")

    colours = ("#4fc3f7", "#c792ea", "#ffb26b", "#66d9a3")
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    for index, (label, root, seeds) in enumerate(arms):
        colour = colours[index % len(colours)]
        stage2, stage3 = curriculum_bounds(seeds, root)

        # --- evaluation: the curve that answers the question ---------------
        for panel, key, title, scale in (
            (axes[0][0], "completed", "evaluation: episodes completed", 1.0),
            (axes[0][1], "mean_distance_m", "evaluation: mean error (cm)", 100.0),
        ):
            stacked = []
            for seed in seeds:
                steps = column(seed["evaluation"], "timesteps")
                values = column(seed["evaluation"], key) * scale
                panel.plot(steps, values, color=colour, alpha=0.25, linewidth=1)
                stacked.append(values)
            width = min(len(v) for v in stacked)
            mean = np.nanmean(np.array([v[:width] for v in stacked]), axis=0)
            panel.plot(column(seeds[0]["evaluation"], "timesteps")[:width], mean,
                       color=colour, linewidth=2.2, label=f"{label} (n={len(seeds)})")
            panel.set_title(title, fontsize=10)
            panel.set_xlabel("timesteps")
            panel.legend(fontsize=8)

        # --- training: harder over time, so shade the curriculum -----------
        for panel, source, key, title, scale in (
            (axes[1][0], "monitor", "r", "training: episode reward (smoothed)", 1.0),
            (axes[1][1], "episodes", "mean_distance_m",
             "training: mean error, cm (smoothed)", 100.0),
        ):
            for seed in seeds:
                rows = seed[source]
                if not rows:
                    continue
                values = column(rows, key) * scale
                steps = (
                    column(rows, "timesteps") if source == "episodes"
                    else np.linspace(0, stage3 / config.CURRICULUM_RANDOMIZED_END,
                                     len(values))
                )
                finite = np.isfinite(values)
                panel.plot(steps[finite], smooth(values[finite], args.smooth),
                           color=colour, alpha=0.55, linewidth=1.2)
            panel.set_title(title, fontsize=10)
            panel.set_xlabel("timesteps")

        for panel in (axes[1][0], axes[1][1]):
            panel.axvspan(0, stage2, color="#66d9a3", alpha=0.06)
            panel.axvspan(stage2, stage3, color="#ffce6b", alpha=0.06)
            panel.axvspan(stage3, panel.get_xlim()[1], color="#ff7a7a", alpha=0.06)

    axes[1][0].annotate("nominal | randomised | disturbances",
                        xy=(0.02, 0.04), xycoords="axes fraction", fontsize=8,
                        color="#93a1b8")
    figure.suptitle(
        "Training and evaluation curves. Evaluation runs a fixed suite, so a flat "
        "tail means converged.\nTraining error rises because the curriculum makes "
        "episodes harder, not because the policy gets worse.",
        fontsize=9,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=130)
    plt.close(figure)

    print(f"\n{'arm':>16}{'evals':>7}{'final done':>12}{'last-quarter change':>22}"
          f"{'seed spread':>13}")
    for label, _root, seeds in arms:
        done = convergence(seeds, "completed")
        dist = convergence(seeds, "mean_distance_m")
        if not done:
            continue
        print(f"{label:>16}{len(seeds[0]['evaluation']):>7}"
              f"{done['final']:>12.1f}{done['change']:>+22.2f}"
              f"{done['seed_spread']:>13.2f}")
        verdict = (
            "PLATEAUED -- the remaining change is smaller than the difference "
            "between seeds"
            if abs(done["change"]) <= max(done["seed_spread"], 0.5)
            else "STILL MOVING -- more training may help"
        )
        print(f"{'':>16}completion {verdict}")
        if dist:
            print(f"{'':>16}mean error changed {dist['change'] * 100:+.3f} cm over the "
                  f"final quarter (seed spread {dist['seed_spread'] * 100:.3f} cm)")

    print(f"\nSaved {args.output}")
    print(
        "\nReading the figure: the top row is the deterministic evaluation suite and\n"
        "is the evidence for convergence. The bottom row is training, where error\n"
        "rises because the curriculum introduces randomised physics at "
        f"{config.CURRICULUM_NOMINAL_END:.0%} and\ndisturbances at "
        f"{config.CURRICULUM_RANDOMIZED_END:.0%} of the budget -- shaded green, amber, red."
    )


if __name__ == "__main__":
    main()
