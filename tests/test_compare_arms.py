from __future__ import annotations

import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import compare_arms


COLUMNS = [
    "scenario", "path_key", "v_target", "mass", "friction",
    "actuator_delay_s", "sensor_noise_m", "event", "controller",
    "finished", "mean_distance_m", "mean_kp",
]


def write_confirmation(
    path: Path,
    ppo_finished: list[int],
    fixed_finished: list[int],
    kps: list[float] | None = None,
    delays: list[float] | None = None,
) -> None:
    """A synthetic confirm_advantage.py CSV with the columns the loader reads."""
    count = len(ppo_finished)
    kps = kps or [30.0] * count
    delays = delays or [0.01 * (index % 5) for index in range(count)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for index in range(count):
            for name, finished, kp in (
                ("PPO", ppo_finished[index], kps[index]),
                ("fixed Kp 26 (baseline)", fixed_finished[index], 26.0),
            ):
                writer.writerow([
                    index, "arc", 0.5, 30.0, 0.5, f"{delays[index]:.4f}", 0.0002,
                    "none", name, finished, 0.01 + 0.001 * index, kp,
                ])


def build_arm(root: Path, seeds: list[int], ppo: list[int], fixed: list[int], **kwargs) -> Path:
    for seed in seeds:
        write_confirmation(root / f"seed{seed}" / "confirm_200.csv", ppo, fixed, **kwargs)
    return root


class ArmSpecTests(unittest.TestCase):
    def test_a_named_arm_keeps_its_label(self):
        arm = compare_arms.parse_arm("context=runs/rq2_context")
        self.assertEqual(arm.name, "context")
        self.assertEqual(arm.root, Path("runs/rq2_context"))

    def test_a_bare_path_is_labelled_by_its_directory(self):
        arm = compare_arms.parse_arm("runs/rq1_blind")
        self.assertEqual(arm.name, "rq1_blind")


class SuiteIntegrityTests(unittest.TestCase):
    """The fixed-gain rows involve no network, so across arms they must match.

    If they do not, the arms ran different scenarios and the comparison would
    measure the suite rather than the observation.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def load(self, *roots: Path) -> list[compare_arms.Arm]:
        arms = [compare_arms.Arm(root.name, root) for root in roots]
        for arm in arms:
            arm.load("confirm_200.csv")
        return arms

    def test_matching_fixed_rows_pass(self):
        blind = build_arm(self.root / "blind", [7], [1, 0, 1, 1], [1, 0, 0, 1])
        context = build_arm(self.root / "context", [7], [1, 1, 1, 1], [1, 0, 0, 1])
        message = compare_arms.check_the_arms_ran_the_same_suite(self.load(blind, context))
        self.assertIn("2/4", message)

    def test_a_different_fixed_result_is_refused(self):
        blind = build_arm(self.root / "blind", [7], [1, 0, 1, 1], [1, 0, 0, 1])
        context = build_arm(self.root / "context", [7], [1, 1, 1, 1], [1, 1, 1, 1])
        with self.assertRaises(SystemExit) as caught:
            compare_arms.check_the_arms_ran_the_same_suite(self.load(blind, context))
        self.assertIn("scenario suite", str(caught.exception))

    def test_tracking_is_read_only_on_episodes_every_arm_finished(self):
        blind = build_arm(self.root / "blind", [7], [1, 0, 1, 1], [1, 0, 0, 1])
        context = build_arm(self.root / "context", [7], [1, 1, 1, 0], [1, 0, 0, 1])
        self.assertEqual(compare_arms.common_finished(self.load(blind, context), 7), [0, 2])


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def run_main(self, *argv: str) -> str:
        buffer = io.StringIO()
        with mock.patch("sys.argv", ["compare_arms.py", *argv]):
            with contextlib.redirect_stdout(buffer):
                compare_arms.main()
        return buffer.getvalue()

    def test_one_arm_is_refused(self):
        build_arm(self.root / "blind", [7], [1, 1], [1, 0])
        with self.assertRaises(SystemExit):
            self.run_main(f"--arm=blind={self.root / 'blind'}")

    def test_arms_with_no_seed_in_common_are_refused(self):
        build_arm(self.root / "blind", [7], [1, 1], [1, 0])
        build_arm(self.root / "context", [21], [1, 1], [1, 0])
        with self.assertRaises(SystemExit) as caught:
            self.run_main(f"--arm=blind={self.root / 'blind'}",
                          f"--arm=context={self.root / 'context'}")
        self.assertIn("no seed in common", str(caught.exception))

    def test_a_half_finished_batch_is_refused_until_asked_for(self):
        ppo, fixed = [1, 0, 1, 1], [1, 0, 0, 1]
        build_arm(self.root / "blind", [7, 21], ppo, fixed)
        build_arm(self.root / "context", [7], ppo, fixed)
        args = (f"--arm=blind={self.root / 'blind'}",
                f"--arm=context={self.root / 'context'}")
        with self.assertRaises(SystemExit) as caught:
            self.run_main(*args)
        self.assertIn("--allow-partial-seeds", str(caught.exception))
        report = self.run_main(*args, "--allow-partial-seeds")
        self.assertIn("WARNING", report)

    def test_a_clean_comparison_reports_both_statistics(self):
        fixed = [1, 0, 0, 1, 0, 1]
        seeds = [7, 21, 42]
        build_arm(self.root / "blind", seeds, [1, 0, 0, 1, 0, 1], fixed,
                  kps=[30.0] * 6)
        # The context arm finishes two episodes the blind one cannot, on every
        # seed, and lowers its gain as dead time rises.
        build_arm(self.root / "context", seeds, [1, 1, 1, 1, 0, 1], fixed,
                  kps=[45.0, 40.0, 35.0, 30.0, 25.0, 20.0],
                  delays=[0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
        report = self.run_main(f"--arm=blind={self.root / 'blind'}",
                               f"--arm=context={self.root / 'context'}")
        self.assertIn("integrity", report)
        self.assertIn("context against blind", report)
        self.assertIn("sign test p = 0.2500", report)  # 3 wins, 0 losses
        self.assertIn("scheduling", report)
        # A perfectly scheduling arm shows -1.00 on the delay column.
        self.assertIn("-1.00", report)

    def test_unequal_scenario_counts_are_refused(self):
        build_arm(self.root / "blind", [7], [1, 0, 1, 1], [1, 0, 0, 1])
        build_arm(self.root / "context", [7], [1, 0, 1], [1, 0, 0])
        with self.assertRaises(SystemExit) as caught:
            self.run_main(f"--arm=blind={self.root / 'blind'}",
                          f"--arm=context={self.root / 'context'}")
        self.assertIn("not paired", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
