from __future__ import annotations

import unittest

import numpy as np

from adaptation_metrics import adaptation_metrics, paired_row
from adaptation_scenarios import EventDefinition, PhysicsConfig, Scenario


def dynamic_scenario() -> Scenario:
    return Scenario(
        "arc_v0p3_force_5",
        "arc",
        0.3,
        7,
        PhysicsConfig(),
        EventDefinition("force_pulse", 2.0, 2.5, 5.0),
        ("dynamic", "force"),
    )


class AdaptationMetricTests(unittest.TestCase):
    def test_known_trace_recovers_after_response_window(self):
        rows = []
        for time_s in np.arange(0.0, 6.02, 0.02):
            error = 0.10 if time_s < 2.0 or time_s >= 3.0 else 0.50
            rows.append({
                "t": time_s,
                "dist": error,
                "kp": 8.0 if time_s < 2.1 else 10.0,
                "ki": 0.0,
                "kd": 0.3,
            })
        metrics = adaptation_metrics(rows, dynamic_scenario())
        self.assertAlmostEqual(metrics["pre_median_error_m"], 0.10)
        self.assertAlmostEqual(metrics["post_peak_error_m"], 0.50)
        self.assertAlmostEqual(metrics["recovery_threshold_m"], 0.145)
        self.assertAlmostEqual(metrics["recovery_time_s"], 2.0, places=6)
        self.assertFalse(metrics["failed_to_recover"])
        self.assertAlmostEqual(metrics["gain_response_time_s"], 0.1, places=6)

    def test_failure_to_recover_is_explicit(self):
        rows = [
            {"t": time_s, "dist": 0.1 if time_s < 2.0 else 0.5, "kp": 8.0, "ki": 0.0, "kd": 0.3}
            for time_s in np.arange(0.0, 5.0, 0.02)
        ]
        metrics = adaptation_metrics(rows, dynamic_scenario())
        self.assertTrue(metrics["failed_to_recover"])
        self.assertIsNone(metrics["recovery_time_s"])

    def test_pair_sign_is_ppo_minus_fixed(self):
        scenario = dynamic_scenario()
        fixed = {"finished": True, "mean_distance_m": 0.2}
        ppo = {"finished": True, "mean_distance_m": 0.15}
        row = paired_row(scenario, fixed, ppo)
        self.assertAlmostEqual(row["delta_mean_distance_m"], -0.05)
        self.assertEqual(row["delta_sign"], "ppo_minus_fixed")
        self.assertEqual(row["disturbance_family"], "force_pulse")
        self.assertEqual(row["severity_value"], 5.0)


if __name__ == "__main__":
    unittest.main()
