from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

import config
from calibration import GainCalibration
from core.dynamics import DT
from dashboard import (
    DirectComparisonRunner,
    comparison_view,
    create_app,
    deviation_figure,
    gain_controls,
    resolve_calibration,
    trajectory_figure,
)
from env import PathFollowingEnv


def _require_model(project: Path) -> None:
    """These tests need a trained PPO artifact. Skip rather than fail when the
    working tree simply has not produced one yet.

    Checks every candidate, not one fixed path: the dashboard now loads a panel
    per model it can find, so pinning this to DEFAULT_MODEL silently skipped
    the whole class whenever the preferred run directory was absent -- which is
    the normal state on a machine that has not copied the batch back.
    """
    if not any(
        (project / path).exists()
        for _label, path in DirectComparisonRunner.MODEL_CANDIDATES
    ):
        raise unittest.SkipTest("no trained model found; run train.py first")


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = Path(__file__).resolve().parents[1]
        _require_model(cls.project)
        cls.runner = DirectComparisonRunner(cls.project)
        cls.result = cls.runner.compare("arc", 0.3)
        cls.fixed = cls.result["fixed"]

    def test_direct_runner_uses_matched_nominal_conditions(self):
        self.assertEqual(self.result["path_key"], "arc")
        self.assertEqual(self.result["target_speed"], 0.3)
        self.assertGreater(len(self.result["fixed"]["trace"]), 0)
        self.assertTrue(all(len(m["trace"]) > 0 for m in self.result["models"]))
        for panel in [self.result["fixed"], *self.result["models"]]:
            first = panel["trace"][0]
            self.assertEqual(first["path_key"], "arc")
            self.assertEqual(first["v_target"], 0.3)
            self.assertEqual(first["base_mass"], 10.0)
            self.assertEqual(first["event_kind"], "none")

    def test_direct_runner_rejects_unlisted_inputs(self):
        with self.assertRaises(ValueError):
            self.runner.compare("../escape", 0.3)
        with self.assertRaises(ValueError):
            self.runner.compare("arc", 9.9)

    def test_trajectory_contains_sandbox_overlay_and_full_run(self):
        trace = self.fixed["trace"]
        figure = trajectory_figure(trace, "arc", "Fixed PID", "#4fc3f7")
        self.assertEqual(figure.data[0].name, "corridor")
        self.assertEqual(figure.data[1].name, "reference path")
        self.assertEqual(figure.data[2].name, "Fixed PID")
        self.assertEqual(len(figure.data[2].x), len(trace))
        self.assertEqual(figure.data[3].name, "start")
        self.assertEqual(figure.data[4].name, "finish")
        self.assertEqual(figure.layout.yaxis.scaleanchor, "x")

    def test_deviation_figure_shows_controller_and_true_error(self):
        trace = self.fixed["trace"]
        figure = deviation_figure(trace)
        self.assertEqual(
            {item.name for item in figure.data},
            {"distance off path", "e_ct (controller)"},
        )
        self.assertEqual(len(figure.data[0].x), len(trace))

    def test_view_is_only_the_two_controller_columns(self):
        view = comparison_view(self.result, self.runner)
        self.assertEqual(view.className, "comparison-grid")
        self.assertEqual(len(view.children), 2)
        self.assertEqual(view.children[0].children[0].children, "Fixed PID")
        self.assertEqual(view.children[1].children[0].children, "PPO")

    def test_app_has_only_path_and_speed_inputs(self):
        app = create_app(self.project)
        client = app.server.test_client()
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        layout_text = str(app.layout)
        self.assertIn("path-select", layout_text)
        self.assertIn("speed-select", layout_text)
        self.assertNotIn("run-select", layout_text)


class GustDisturbanceTests(unittest.TestCase):
    """The gust is only usable as evidence if it is reproducible and calibrated."""

    GUST = {
        "kind": "force_gust",
        "start_s": 0.0,
        "end_s": 30.0,
        "amount": 10.0,
        "tau_s": 0.05,
        "seed": config.SEED,
    }

    @classmethod
    def setUpClass(cls):
        cls.project = Path(__file__).resolve().parents[1]
        # Loaded straight from the artifact: the OU gust is a property of the
        # environment, so these tests should not need a trained policy to exist.
        cls.calibration = GainCalibration.load(resolve_calibration(cls.project))

    def _armed_env(self) -> PathFollowingEnv:
        env = PathFollowingEnv(
            calibration=self.calibration, training=False, path_keys=("arc",)
        )
        env.reset(
            seed=config.SEED,
            options={"path_key": "arc", "v_target": 0.3, "disturbances": [dict(self.GUST)]},
        )
        return env

    def test_gust_spread_matches_the_requested_strength(self):
        env = self._armed_env()
        try:
            for _ in range(5000):  # burn in from the calm start
                env._advance_gust(0, self.GUST)
            samples = np.array(
                [env._advance_gust(0, self.GUST) for _ in range(100000)]
            )
        finally:
            env.close()
        self.assertAlmostEqual(
            float(samples.std()), self.GUST["amount"], delta=0.08 * self.GUST["amount"]
        )
        # Both axes wander independently, so the bearing covers the compass
        # instead of flipping along one line.
        self.assertLess(abs(float(np.corrcoef(samples[:, 0], samples[:, 1])[0, 1])), 0.05)

    def test_correlation_time_sets_how_fast_the_wind_veers(self):
        env = self._armed_env()
        try:
            for _ in range(5000):
                env._advance_gust(0, self.GUST)
            series = np.array([env._advance_gust(0, self.GUST)[0] for _ in range(200000)])
        finally:
            env.close()
        lag = int(round(float(self.GUST["tau_s"]) / DT))
        measured = float(np.corrcoef(series[:-lag], series[lag:])[0, 1])
        self.assertAlmostEqual(measured, math.exp(-1.0), delta=0.05)

    def test_both_controllers_meet_the_same_wind(self):
        first, second = self._armed_env(), self._armed_env()
        try:
            a = np.array([first._advance_gust(0, self.GUST) for _ in range(500)])
            b = np.array([second._advance_gust(0, self.GUST) for _ in range(500)])
        finally:
            first.close()
            second.close()
        np.testing.assert_array_equal(a, b)

    def test_gust_reaches_the_body_only_inside_its_window(self):
        env = self._armed_env()
        try:
            from core.dynamics import car_id

            env.time_s = 40.0  # past end_s
            env._apply_disturbance()
            np.testing.assert_array_equal(env.data.xfrc_applied[car_id(), 0:2], [0.0, 0.0])
            env.time_s = 1.0
            env._apply_disturbance()
            self.assertGreater(float(np.abs(env.data.xfrc_applied[car_id(), 0:2]).sum()), 0.0)
        finally:
            env.close()

    def test_runner_threads_the_gust_through_and_it_changes_the_run(self):
        _require_model(self.project)
        runner = DirectComparisonRunner(self.project)
        calm = runner.compare("arc", 0.3)
        windy = runner.compare(
            "arc", 0.3, gust_enabled=True, gust=20.0, gust_tau=1.0,
            gust_start=0.0, gust_end=30.0,
        )
        self.assertEqual([event["kind"] for event in windy["events"]], ["force_gust"])
        self.assertNotAlmostEqual(
            calm["fixed"]["metrics"]["mean_distance_m"],
            windy["fixed"]["metrics"]["mean_distance_m"],
        )

    def test_runner_rejects_an_empty_gust_window(self):
        _require_model(self.project)
        runner = DirectComparisonRunner(self.project)
        with self.assertRaises(ValueError):
            runner.compare("arc", 0.3, gust_enabled=True, gust_start=5.0, gust_end=5.0)


class InteractiveTuningTests(unittest.TestCase):
    """The hand-tuning panel, which must work BEFORE a model exists.

    That is the whole point of the degraded mode: the left panel is useful for
    tuning against the live plant while training has not been run yet, and the
    dashboard used to refuse to start at all without a model artifact.
    """

    @classmethod
    def setUpClass(cls):
        cls.project = Path(__file__).resolve().parents[1]
        cls.runner = DirectComparisonRunner(cls.project)

    def test_the_dashboard_starts_without_a_trained_model(self):
        if self.runner.models:
            self.skipTest("a model exists, so the degraded path cannot be exercised")
        result = self.runner.compare("arc", 0.3)
        self.assertEqual(result["models"], [])
        self.assertIn("train.py", self.runner.model_error)
        view = comparison_view(result, self.runner)
        self.assertIn("PPO", str(view))
        self.assertIsNotNone(create_app(self.project))

    def test_gain_sliders_actually_change_the_fixed_pid_run(self):
        """And under dead time plus noise, the HIGHER gain is the worse one --
        the same inversion GainTradeoffTests asserts, reachable by hand here."""
        low = self.runner.compare(
            "arc", 0.5, delay_ms=60.0, noise_mm=0.3, kp=25.0
        )["fixed"]["metrics"]["mean_distance_m"]
        high = self.runner.compare(
            "arc", 0.5, delay_ms=60.0, noise_mm=0.3, kp=50.0
        )["fixed"]["metrics"]["mean_distance_m"]
        self.assertNotAlmostEqual(low, high)
        self.assertLess(low, high)

    def test_the_imperfection_sliders_reach_the_plant(self):
        clean = self.runner.compare("arc", 0.5, delay_ms=0.0, noise_mm=0.0)
        rough = self.runner.compare("arc", 0.5, delay_ms=100.0, noise_mm=0.5)
        self.assertEqual(clean["delay_s"], 0.0)
        self.assertAlmostEqual(rough["delay_s"], 0.1)
        self.assertAlmostEqual(rough["noise_m"], 0.0005)
        for row in clean["fixed"]["trace"]:
            self.assertEqual(row["e_ct_measured"], row["e_ct"])
        self.assertTrue(
            any(
                row["e_ct_measured"] != row["e_ct"]
                for row in rough["fixed"]["trace"]
            )
        )

    def test_moving_a_gain_does_not_rerun_the_ppo_episode(self):
        """The PPO side does not depend on the Fixed PID's gains, so it must be
        cached against the scenario alone. Sharing one key would re-run the
        slow controller on every drag of a slider."""
        runner = DirectComparisonRunner(self.project)
        runner.compare("arc", 0.3, kp=20.0)
        runner.compare("arc", 0.3, kp=40.0)
        self.assertEqual(len(runner._fixed_cache), 2)
        self.assertLessEqual(len(runner._model_cache), len(runner.models))

    def test_gain_defaults_are_the_calibrated_baseline(self):
        defaults = [control[5] for control in gain_controls(self.runner.calibration)]
        np.testing.assert_allclose(defaults, self.runner.calibration.base)

    def test_gains_outside_the_search_space_are_rejected(self):
        with self.assertRaises(ValueError):
            self.runner.compare("arc", 0.3, kp=500.0)
        with self.assertRaises(ValueError):
            self.runner.compare("arc", 0.3, kd=-1.0)

    def test_a_named_but_missing_model_is_an_error_not_a_downgrade(self):
        with self.assertRaises(FileNotFoundError):
            DirectComparisonRunner(self.project, "runs/does_not_exist.zip")

    def test_a_model_whose_arms_manifest_is_wrong_is_rejected(self):
        """A missing arms.json defaults to blind, which is correct for runs made
        before train.py wrote one -- and silently wrong for a context model that
        lost its manifest. That must be caught at load, not inside SB3."""
        if not self.runner.models:
            self.skipTest("no model available")
        source = self.runner.models[0]["path"]
        with tempfile.TemporaryDirectory(dir=self.project / "runs") as directory:
            staged = Path(directory) / "best_model.zip"
            shutil.copy(source, staged)
            # Claim the blind model needs the wider context observation.
            (Path(directory) / "arms.json").write_text(
                json.dumps({"preview": True, "plant_context": True}), encoding="utf-8"
            )
            relative = staged.relative_to(self.project)
            with self.assertRaises(ValueError) as caught:
                DirectComparisonRunner(self.project, str(relative))
            self.assertIn("arms.json", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
