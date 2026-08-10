"""Launch several training seeds at once, one process per seed.

Why one process per seed rather than parallel environments inside one run:
measured on this project, env stepping is 47% of training wall time and the PPO
gradient update is the other 53%, and that half is serial. So env parallelism
caps at about 1.7x however many cores it gets, while independent seeds scale
close to linearly. Five seeds on five cores beats one seed on five cores by
roughly 3x, and -- more importantly for a thesis batch -- it leaves each run's
training dynamics byte-for-byte what was validated on a single core.

Each child gets a single-threaded BLAS/torch. Without that every process grabs
every core and they fight: five runs x 32 threads on 32 cores is slower than
five single-threaded runs.

    python run_seeds.py --seeds 7,21,42,84,123 --total-timesteps 1000000 \
        --calibration runs/calibration_v7.json --run-root runs/rq1_blind
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parent


def usable_cores() -> int:
    """Cores this process may actually run on, not the ones the box has.

    Under SLURM (or any cgroup/affinity limit) `os.cpu_count()` reports the
    whole compute node -- 128 on a machine where the allocation is 6 -- so a
    batch sized from it looks healthy and then crawls, four processes deep on
    every core. The affinity mask is what the scheduler actually granted.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return os.cpu_count() or 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="7,21,42,84,123")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Concurrent processes. 0 means one per seed, all at once.",
    )
    parser.add_argument(
        "--threads-per-job",
        type=int,
        default=1,
        help=(
            "Torch threads inside each run. Keep at 1 unless the batch is "
            "smaller than the core count and you want the spare cores used."
        ),
    )
    parser.add_argument(
        "--extra",
        type=str,
        default="",
        help=(
            "Passed through to train.py verbatim, e.g. \"--no-delay --no-noise\" "
            "for the clean-plant ablation arm. Recorded in the manifest."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    if not seeds:
        raise SystemExit("no seeds given")
    calibration = (PROJECT / args.calibration).resolve()
    if not calibration.exists():
        raise SystemExit(f"calibration not found: {calibration}")

    run_root = PROJECT / args.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    extra = args.extra.split()
    jobs = args.jobs if args.jobs > 0 else len(seeds)

    # The manifest is what makes a batch auditable months later: which seeds,
    # which artifact, which flags, which machine.
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "total_timesteps": args.total_timesteps,
        "calibration": str(calibration.relative_to(PROJECT)),
        "extra_args": extra,
        "threads_per_job": args.threads_per_job,
        "concurrent_jobs": jobs,
        "usable_cores": usable_cores(),
        "cpu_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    (run_root / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))

    child_env = dict(os.environ)
    # Belt and braces: torch honours --torch-threads, but numpy's BLAS and
    # OpenMP read these and would otherwise thread out across every core.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        child_env[name] = str(args.threads_per_job)
    # MUJOCO_GL is deliberately NOT set here. Training never renders, and
    # mujoco only needs a GL backend when a rendering context is actually
    # created -- so importing it headless is fine with the variable unset,
    # while setting it to a backend the machine lacks fails the import outright
    # (setting `egl` on Windows raises at `import mujoco`). If a server does
    # complain about GL, export MUJOCO_GL=osmesa in the shell rather than here.

    pending = list(seeds)
    running: list[tuple[int, subprocess.Popen, Path, float]] = []
    finished: list[tuple[int, int, float]] = []

    def launch(seed: int) -> None:
        run_dir = run_root / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(PROJECT / "train.py"),
            "--calibration", str(calibration),
            "--total-timesteps", str(args.total_timesteps),
            "--seed", str(seed),
            "--run-dir", str(run_dir),
            "--torch-threads", str(args.threads_per_job),
            *extra,
        ]
        log_path = run_dir / "train.log"
        print(f"  launching seed {seed} -> {run_dir}  (log: {log_path})")
        if args.dry_run:
            print("    " + " ".join(command))
            return
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command, cwd=str(PROJECT), env=child_env,
            stdout=handle, stderr=subprocess.STDOUT,
        )
        running.append((seed, process, log_path, time.time()))

    cores = usable_cores()
    wanted = jobs * args.threads_per_job
    print(f"\n{len(seeds)} seeds, {jobs} at a time, "
          f"{args.threads_per_job} thread(s) each = {wanted} threads "
          f"on {cores} usable core(s)\n")
    if wanted > cores:
        print(
            f"  WARNING: asking for {wanted} threads with only {cores} cores "
            "available. Runs will contend and each will be slower than it "
            "would be alone. Lower --jobs, or request more cores from the "
            "scheduler.\n"
        )
    while pending or running:
        while pending and len(running) < jobs:
            launch(pending.pop(0))
        if args.dry_run:
            break
        time.sleep(5)
        for entry in list(running):
            seed, process, _log, started = entry
            code = process.poll()
            if code is None:
                continue
            running.remove(entry)
            minutes = (time.time() - started) / 60.0
            finished.append((seed, code, minutes))
            state = "ok" if code == 0 else f"FAILED (exit {code})"
            print(f"  seed {seed} {state} after {minutes:.1f} min "
                  f"[{len(finished)}/{len(seeds)} done]")

    if args.dry_run:
        return
    failures = [seed for seed, code, _ in finished if code != 0]
    print("\nbatch complete")
    for seed, code, minutes in sorted(finished):
        print(f"  seed {seed:>4}: exit {code}, {minutes:.1f} min")
    if failures:
        raise SystemExit(f"seeds failed: {failures} -- check seedN/train.log")
    print(f"\nAll {len(seeds)} runs finished. Next: confirm_advantage.py per seed.")


if __name__ == "__main__":
    main()
