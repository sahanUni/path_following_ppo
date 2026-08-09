from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from calibration import GainCalibration
from callbacks import evaluation_conditions
import config
from env import PathFollowingEnv
from rollout import fixed_policy, run_episode


class EvaluationTierTests(unittest.TestCase):
    """Checkpoint selection has to be able to SEE the axis the gain controls.

    Run v7_seed7 sat flat for all 40 checkpoints and froze `best_model` at 50k
    steps, not because nothing improved but because every episode in the suite
    completed at every reachable gain. The suite could only score tracking, and
    on nominal conditions the entire gap from the baseline gains to the
    per-condition best is worth under 8% of episode reward -- often under 2%.
    """

    def test_the_suite_has_a_tier_where_finishing_is_in_doubt(self):
        names = [name for name, _, _ in evaluation_conditions(config.SEED)]
        self.assertIn("stress", names)

    def test_the_stress_tier_actually_changes_the_plant(self):
        conditions = {name: overrides for name, overrides, _ in evaluation_conditions(7)}
        self.assertEqual(conditions["calm"], {})
        stress = conditions["stress"]
        self.assertLess(stress["friction"], config.NOMINAL_FRICTION)
        self.assertGreater(stress["mass"], config.NOMINAL_MASS_KG)
        self.assertGreater(stress["actuator_delay_s"], 0.0)
        self.assertGreater(stress["sensor_noise_m"], 0.0)

    def test_the_gust_tier_is_seeded_so_checkpoints_meet_the_same_wind(self):
        first = evaluation_conditions(11)[1][2][0]
        second = evaluation_conditions(11)[1][2][0]
        self.assertEqual(first, second)
        self.assertEqual(first["seed"], 11)
        self.assertNotEqual(evaluation_conditions(12)[1][2][0]["seed"], 11)


class StressTierIsGainSensitiveTests(unittest.TestCase):
    """The tier must be hard AND discriminating, which are not the same thing.

    Tiers that merely raise mass or dead time were measured and rejected: they
    failed 2 of 9 episodes at EVERY gain from Kp 15 to 50, which is score with
    no information in it. The same trap already forced GUST_SIGMA_RANGE_N down
    to 12 N and EVAL_GUST_SIGMA_N down to 6 N.
    """

    CALIBRATION = Path(__file__).resolve().parents[1] / "runs" / "calibration_v7.json"
    PATHS = ("hairpin", "square", "zigzag30")

    @classmethod
    def setUpClass(cls):
        if not cls.CALIBRATION.exists():
            raise unittest.SkipTest(f"no {cls.CALIBRATION.name}; run calibrate_pid.py")
        cls.calibration = GainCalibration.load(cls.CALIBRATION)
        cls.stress = dict(evaluation_conditions(config.SEED)[2][1])

    def _completions(self, kp: float) -> int:
        gains = self.calibration.base.copy()
        gains[0] = kp
        done = 0
        for path_key in self.PATHS:
            for speed in config.SPEED_TARGETS:
                env = PathFollowingEnv(
                    calibration=self.calibration,
                    training=False,
                    path_keys=(path_key,),
                    fixed_gains=gains,
                )
                try:
                    metrics, _ = run_episode(
                        env,
                        fixed_policy,
                        seed=config.SEED,
                        options={
                            "path_key": path_key,
                            "v_target": speed,
                            "mass": config.NOMINAL_MASS_KG,
                            "friction": config.NOMINAL_FRICTION,
                            "actuator": config.NOMINAL_ACTUATOR,
                            "noise_seed": config.SEED,
                            "stage": 3,
                            "disturbances": [
                                {"kind": "none", "start_s": 0.0,
                                 "end_s": None, "amount": 0.0}
                            ],
                            **self.stress,
                        },
                    )
                finally:
                    env.close()
                done += int(metrics["finished"])
        return done

    def test_a_good_gain_finishes_far_more_than_a_bad_one(self):
        gentle = self._completions(float(self.calibration.low[0]))
        harsh = self._completions(float(self.calibration.high[0]))
        # Measured 9/9 at the box floor against 6/9 at the ceiling.
        self.assertGreater(gentle, harsh)
        # Not unwinnable: if the best reachable gain cannot finish most of it,
        # the tier supplies punishment rather than signal.
        self.assertGreaterEqual(gentle, 8)


if __name__ == "__main__":
    unittest.main()
