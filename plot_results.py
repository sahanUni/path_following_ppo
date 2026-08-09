"""Create trajectory, error, gain, and actuator plots from evaluation CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _read(path: Path) -> dict[int, list[dict]]:
    episodes: dict[int, list[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episodes[int(row["episode_id"])].append(row)
    return dict(episodes)


def _float(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def plot_episode(rows: list[dict], output: Path) -> None:
    t = _float(rows, "t")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(_float(rows, "x"), _float(rows, "y"), color="#2563eb")
    axes[0, 0].set(title="Vehicle trajectory", xlabel="x [m]", ylabel="y [m]")
    axes[0, 0].axis("equal")

    axes[0, 1].plot(t, _float(rows, "dist"), label="true distance")
    axes[0, 1].plot(t, [abs(x) for x in _float(rows, "e_ct")], label="|e_ct|", alpha=0.75)
    axes[0, 1].set(title="Tracking error", xlabel="time [s]", ylabel="error [m]")
    axes[0, 1].legend()

    for gain in ("kp", "ki", "kd"):
        axes[1, 0].plot(t, _float(rows, gain), label=gain)
    axes[1, 0].set(title="Applied steering PID gains", xlabel="time [s]")
    axes[1, 0].legend()

    axes[1, 1].plot(t, _float(rows, "u_left"), label="left wheel")
    axes[1, 1].plot(t, _float(rows, "u_right"), label="right wheel")
    axes[1, 1].axhline(1.0, color="black", linestyle=":", linewidth=1)
    axes[1, 1].axhline(-1.0, color="black", linestyle=":", linewidth=1)
    axes[1, 1].set(title="Applied wheel commands", xlabel="time [s]")
    axes[1, 1].legend()

    first = rows[0]
    fig.suptitle(
        f"{first['controller']} · {first['path_key']} · v*={float(first['v_target']):.1f} m/s"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_trace_file(trace_path: Path, output: Path, episode_id: int | None = None) -> None:
    episodes = _read(trace_path)
    selected = [episode_id] if episode_id is not None else sorted(episodes)
    for current in selected:
        if current not in episodes:
            raise ValueError(f"episode {current} is not present in {trace_path}")
        destination = output.with_name(f"{output.stem}_{current:03d}{output.suffix}")
        plot_episode(episodes[current], destination)
        print(f"Saved {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_csv", type=Path)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation/episode.png"))
    args = parser.parse_args()
    plot_trace_file(args.trace_csv, args.output, args.episode)


if __name__ == "__main__":
    main()

