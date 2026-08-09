from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from calibration import GainCalibration
import config
from calibrate_pid import (
    assert_two_sided,
    derive_calibration,
    grid_candidates,
    rank_key,
    rank_rows,
)


class CalibrationTests(unittest.TestCase):
    def test_completion_is_ranked_before_distance(self):
        complete_but_wider = {
            "kp": 4.0, "ki": 0.0, "kd": 0.0,
            "completed": 3, "mean_distance_m": 0.2, "mean_itae": 1.0,
        }
        incomplete_but_narrow = {
            "kp": 1.0, "ki": 0.0, "kd": 0.0,
            "completed": 2, "mean_distance_m": 0.01, "mean_itae": 0.1,
        }
        self.assertLess(rank_key(complete_but_wider), rank_key(incomplete_but_narrow))

    def test_derived_bounds_contain_baseline_and_have_nonzero_span(self):
        rows = [
            {"kp": 4.0, "ki": 0.0, "kd": 0.0, "completed": 3,
             "mean_distance_m": 0.1, "mean_itae": 1.0},
            {"kp": 8.0, "ki": 0.0, "kd": 0.3, "completed": 3,
             "mean_distance_m": 0.05, "mean_itae": 0.7},
        ]
        calibration = derive_calibration(rows, {})
        self.assertTrue(np.all(calibration.low <= calibration.base))
        self.assertTrue(np.all(calibration.base <= calibration.high))
        self.assertTrue(np.all(calibration.high > calibration.low))

    def test_winning_candidate_at_the_edge_still_leaves_room_to_raise(self):
        """The exact shape that welded Kp shut: the best row is also the most
        extreme survivor, so max(candidate gains) collapses onto the baseline."""
        rows = [
            {"kp": 42.0, "ki": 2.0, "kd": 4.0, "completed": 3,
             "mean_distance_m": 0.01, "mean_itae": 0.5},
            {"kp": 28.0, "ki": 2.9, "kd": 4.3, "completed": 3,
             "mean_distance_m": 0.02, "mean_itae": 0.6},
        ]
        calibration = derive_calibration(rows, {})
        self.assertGreater(calibration.high[0], calibration.base[0])
        self.assertLess(calibration.low[0], calibration.base[0])

    def test_every_action_channel_moves_its_gain_both_ways(self):
        rows = [
            {"kp": 42.0, "ki": 2.0, "kd": 4.0, "completed": 3,
             "mean_distance_m": 0.01, "mean_itae": 0.5},
            {"kp": 28.0, "ki": 2.9, "kd": 4.3, "completed": 3,
             "mean_distance_m": 0.02, "mean_itae": 0.6},
        ]
        calibration = derive_calibration(rows, {})
        for index in range(3):
            self.assertLess(calibration.low[index], calibration.base[index])
            self.assertGreater(calibration.high[index], calibration.base[index])

    def test_a_welded_channel_is_rejected(self):
        welded = GainCalibration(
            base=np.array([42.0, 2.0, 4.0]),
            low=np.array([26.0, 1.5, 1.8]),
            high=np.array([42.0, 2.9, 4.3]),  # Kp cannot be raised
            metadata={},
        )
        with self.assertRaises(ValueError) as caught:
            assert_two_sided(welded)
        self.assertIn("Kp", str(caught.exception))

    def test_a_channel_pinned_at_the_search_limit_is_allowed(self):
        """Ki of zero has nowhere lower to go; that is physics, not a bug."""
        pinned = GainCalibration(
            base=np.array([42.0, config.CALIBRATION_SEARCH_LOW[1], 4.0]),
            low=np.array([26.0, config.CALIBRATION_SEARCH_LOW[1], 1.8]),
            high=np.array([50.0, 2.9, 5.2]),
            metadata={},
        )
        assert_two_sided(pinned)

    def test_the_shipped_artifact_has_no_welded_channel(self):
        path = Path(__file__).resolve().parents[1] / "runs" / "calibration_twosided.json"
        assert_two_sided(GainCalibration.load(path))


class CompletionBandTests(unittest.TestCase):
    """A gain that only survives by crawling must not win on completion alone.

    Taken from the v6 sweep, where Kp 1.23 completed 99/99 at 5.25 cm and was
    selected over Kp 20.32 at 93/99 and 1.88 cm -- which beat it head to head on
    all 93 shared scenarios. The six extra completions were one adverse physics
    draw on which the crawler wandered 9-29 cm inside a 1.0 m corridor.
    """

    CRAWLER = {
        "kp": 1.23, "ki": 0.03, "kd": 2.46,
        "completed": 99, "mean_distance_m": 0.0525, "mean_itae": 14.3,
    }
    TRACKER = {
        "kp": 20.32, "ki": 0.0, "kd": 4.38,
        "completed": 93, "mean_distance_m": 0.0188, "mean_itae": 8.1,
    }
    FRAGILE = {
        "kp": 42.0, "ki": 2.14, "kd": 4.15,
        "completed": 66, "mean_distance_m": 0.0565, "mean_itae": 78.9,
    }

    def test_the_better_tracker_wins_inside_the_completion_band(self):
        ranked = rank_rows([dict(self.CRAWLER), dict(self.TRACKER)])
        self.assertEqual(ranked[0]["kp"], self.TRACKER["kp"])

    def test_completion_still_decides_across_the_big_gap(self):
        """The band is a tie-break, not an abandonment of robustness: a gain
        that fails a third of the suite stays disqualified."""
        tight_but_fragile = dict(self.FRAGILE, mean_distance_m=0.001, mean_itae=0.1)
        ranked = rank_rows([dict(self.CRAWLER), tight_but_fragile])
        self.assertEqual(ranked[0]["kp"], self.CRAWLER["kp"])

    def test_the_band_is_wider_than_one_physics_sample(self):
        """The whole crawler advantage was 6 of 99 scenarios, one physics draw.
        If the tolerance ever tightens below that, the artifact comes back."""
        self.assertLessEqual(config.COMPLETION_TOLERANCE, 1.0 - 6.0 / 99.0)

    def test_the_box_is_bounded_by_everything_in_the_band(self):
        """Not only rows matching the winner's exact count. On a hard plant the
        top count is often reached once, and the box then collapses onto the
        baseline and is rebuilt purely out of MIN_GAIN_MARGIN padding."""
        calibration = derive_calibration(
            [dict(self.CRAWLER), dict(self.TRACKER), dict(self.FRAGILE)], {}
        )
        self.assertEqual(calibration.metadata["box_source"], "derived")
        self.assertLess(calibration.low[0], calibration.base[0])
        self.assertGreater(calibration.high[0], calibration.base[0])


class ExplicitBoxTests(unittest.TestCase):
    ROWS = [
        {"kp": 34.0, "ki": 0.0, "kd": 4.4, "completed": 99,
         "mean_distance_m": 0.006, "mean_itae": 5.0},
        {"kp": 20.0, "ki": 0.0, "kd": 4.4, "completed": 99,
         "mean_distance_m": 0.008, "mean_itae": 6.0},
    ]

    def test_an_explicit_box_is_used_verbatim_and_labelled(self):
        calibration = derive_calibration(
            [dict(r) for r in self.ROWS],
            {},
            box_low=np.array([15.0, 0.0, 2.5]),
            box_high=np.array([50.0, 0.5, 6.0]),
        )
        self.assertEqual(calibration.metadata["box_source"], "explicit")
        np.testing.assert_allclose(calibration.low, [15.0, 0.0, 2.5])
        np.testing.assert_allclose(calibration.high, [50.0, 0.5, 6.0])
        np.testing.assert_allclose(calibration.base, [34.0, 0.0, 4.4])

    def test_an_explicit_box_still_cannot_weld_a_channel(self):
        with self.assertRaises(ValueError):
            derive_calibration(
                [dict(r) for r in self.ROWS],
                {},
                box_low=np.array([15.0, 0.0, 2.5]),
                box_high=np.array([34.0, 0.5, 6.0]),  # Kp high == base
            )

    def test_an_explicit_box_must_stay_inside_the_search_space(self):
        with self.assertRaises(ValueError):
            derive_calibration(
                [dict(r) for r in self.ROWS],
                {},
                box_low=np.array([15.0, 0.0, 2.5]),
                box_high=np.array([500.0, 0.5, 6.0]),
            )

    def test_both_halves_of_the_box_are_required(self):
        with self.assertRaises(ValueError):
            derive_calibration(
                [dict(r) for r in self.ROWS], {}, box_low=np.array([15.0, 0.0, 2.5])
            )


class SweepSurvivesABadBoxTests(unittest.TestCase):
    """A rejected gain box must not destroy the sweep that produced it.

    The candidate scores cost hours; the box is arithmetic over them. Deriving
    the box before writing the CSV meant one mistyped --box-high discarded the
    whole run with nothing on disk.
    """

    def test_candidates_are_on_disk_even_when_the_box_is_rejected(self):
        import calibrate_pid

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "calibration_test.json"
            argv = [
                "calibrate_pid.py",
                "--kp-grid", "20,42",
                "--ki-grid", "0",
                "--kd-grid", "4.4",
                "--paths", "arc",
                "--speeds", "0.5",
                "--physics-samples", "0",
                # Welds Kp shut: high equals whichever candidate wins.
                "--box-low", "15,0,2.5",
                "--box-high", "20,0.5,6.0",
                "--output", str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(ValueError):
                    calibrate_pid.main()
            csv_path = output.with_name(output.stem + "_candidates.csv")
            self.assertTrue(csv_path.exists(), "the sweep results were thrown away")
            rows = calibrate_pid.read_rows(csv_path)
            self.assertEqual(len(rows), 2)
            # And the salvaged file is enough to finish the job for free.
            recovered = derive_calibration(
                rows, {}, np.array([15.0, 0.0, 2.5]), np.array([50.0, 1.5, 6.0])
            )
            self.assertEqual(recovered.metadata["box_source"], "explicit")


class GridCandidateTests(unittest.TestCase):
    def test_the_grid_is_the_full_cross_product(self):
        candidates = grid_candidates([20.0, 42.0], [0.0], [3.5, 4.5])
        self.assertEqual(len(candidates), 4)
        np.testing.assert_allclose(candidates[0], [20.0, 0.0, 3.5])
        np.testing.assert_allclose(candidates[-1], [42.0, 0.0, 4.5])


if __name__ == "__main__":
    unittest.main()

