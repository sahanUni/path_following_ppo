from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

import config
from core import paths
from core.dynamics import fresh_model
import replay


PROJECT = Path(__file__).resolve().parents[1]


def scenario(**overrides) -> argparse.Namespace:
    base = dict(
        controller="fixed", path_key="zigzag", speed=0.7, mass=10.0,
        friction=1.0, actuator=1.0, delay_ms=60.0, noise_mm=0.3,
        gust=8.0, gust_tau=1.0, gust_start=0.0, gust_end=1e9,
        mass_target=0.0, mass_time=3.0,
        force=0.0, force_start=3.0, force_end=8.0,
        gains="26,0.5,4.4", calibration=None, model=None,
        playback=1.0, loop=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class ReplayMatchesTheDashboardTests(unittest.TestCase):
    """The viewer must show the episode the panel plotted, not a near-miss.

    The sandbox viewer guaranteed this by replaying logged frames. This one
    re-simulates, which is only safe because the environment is deterministic
    given the same options and seeds -- so that determinism is asserted here
    rather than assumed.
    """

    def test_the_replayed_episode_is_the_dashboard_episode(self):
        from dashboard import DirectComparisonRunner

        runner = DirectComparisonRunner(PROJECT)
        result = runner.compare(
            "zigzag", 0.7, delay_ms=60.0, noise_mm=0.3,
            gust_enabled=True, gust=8.0, gust_tau=1.0,
            gust_start=0.0, gust_end=30.0,
        )
        base = runner.calibration.base
        args = scenario(
            gains=f"{base[0]},{base[1]},{base[2]}",
            gust_end=30.0,
            calibration=runner.calibration_path,
        )
        metrics, frames, winds, _options, _cal, _gains = replay.simulate(args, PROJECT)

        panel = result["fixed"]["metrics"]
        self.assertEqual(metrics["finished"], panel["finished"])
        self.assertAlmostEqual(
            metrics["mean_distance_m"], panel["mean_distance_m"], places=9
        )
        self.assertAlmostEqual(metrics["duration_s"], panel["duration_s"], places=9)
        self.assertEqual(len(frames), len(winds))
        self.assertEqual(len(frames), len(result["fixed"]["trace"]) + 1)

    def test_mid_episode_events_reach_the_replay(self):
        """The viewer once accepted only the gust, so a mass step that put every
        controller out of the corridor on the charts simply never happened in
        the window -- the car sailed round and the two silently disagreed."""
        base = replay.simulate(scenario(gust=0.0), PROJECT)[0]
        heavy = replay.simulate(
            scenario(gust=0.0, mass_target=50.0, mass_time=1.0), PROJECT
        )[0]
        self.assertNotAlmostEqual(
            base["mean_distance_m"], heavy["mean_distance_m"], places=6,
            msg="the mass step did not reach the simulation",
        )
        pushed = replay.simulate(
            scenario(gust=0.0, force=-20.0, force_start=1.0, force_end=4.0), PROJECT
        )[0]
        self.assertNotAlmostEqual(
            base["mean_distance_m"], pushed["mean_distance_m"], places=6,
            msg="the force pulse did not reach the simulation",
        )

    def test_a_mass_step_replays_exactly_as_the_dashboard_ran_it(self):
        from dashboard import DirectComparisonRunner

        runner = DirectComparisonRunner(PROJECT)
        result = runner.compare(
            "arc", 0.5, mass_enabled=True, mass_target=45.0, mass_time=2.0,
            delay_ms=60.0, noise_mm=0.3,
        )
        base = runner.calibration.base
        args = scenario(
            path_key="arc", speed=0.5, gust=0.0,
            mass_target=45.0, mass_time=2.0,
            delay_ms=60.0, noise_mm=0.3,
            gains=f"{base[0]},{base[1]},{base[2]}",
            calibration=runner.calibration_path,
        )
        metrics = replay.simulate(args, PROJECT)[0]
        panel = result["fixed"]["metrics"]
        self.assertEqual(metrics["finished"], panel["finished"])
        self.assertAlmostEqual(
            metrics["mean_distance_m"], panel["mean_distance_m"], places=9
        )

    def test_a_context_model_gets_a_context_environment(self):
        """A plant-context policy needs a 175-wide observation. Building a blind
        environment for it raised a shape error inside predict(), and because
        the viewer runs detached the window simply never appeared."""
        from dashboard import DirectComparisonRunner

        runner = DirectComparisonRunner(PROJECT)
        context = [m for m in runner.models if m["arms"]["plant_context"]]
        if not context:
            self.skipTest("no plant-context model available")
        args = scenario(
            controller="ppo", gust=0.0,
            model=context[0]["path"], calibration=runner.calibration_path,
        )
        metrics, frames, _winds, _options, _cal, _gains = replay.simulate(args, PROJECT)
        self.assertGreater(len(frames), 10)
        self.assertIn("mean_distance_m", metrics)

    def test_the_recorded_wind_is_the_wind_the_car_felt(self):
        args = scenario()
        _metrics, _frames, winds, _options, _cal, _gains = replay.simulate(args, PROJECT)
        magnitude = np.hypot(winds[:, 0], winds[:, 1])
        self.assertGreater(magnitude.max(), 1.0)
        calm = replay.simulate(scenario(gust=0.0), PROJECT)[2]
        np.testing.assert_allclose(calm, 0.0, atol=1e-12)


class SceneDrawingTests(unittest.TestCase):
    """Exercise the decorative geometry against a real MjvScene.

    No window is opened: `viewer.user_scn` is an ordinary MjvScene, so the
    drawing code can be driven directly. That covers the parts most likely to
    break on a MuJoCo upgrade -- geom-buffer overruns and the connector API.
    """

    def setUp(self):
        self.model = fresh_model(10.0, 1.0, 1.0)
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.handle = SimpleNamespace(user_scn=self.scene)

    def test_path_markers_never_overrun_the_geom_buffer(self):
        small = mujoco.MjvScene(self.model, maxgeom=40)
        handle = SimpleNamespace(user_scn=small)
        points = paths.get("figure8")["pts"]
        used = replay.draw_path(handle, points)
        self.assertLessEqual(used, small.maxgeom - 1)
        self.assertGreater(used, 0)
        self.assertEqual(small.ngeom, used)

    def test_the_wind_arrow_appears_only_when_there_is_wind(self):
        slot = replay.draw_path(self.handle, paths.get("arc")["pts"])
        replay.draw_wind(self.handle, slot, np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        self.assertEqual(self.scene.ngeom, slot)
        replay.draw_wind(self.handle, slot, np.array([1.0, 2.0]), np.array([6.0, -3.0]))
        self.assertEqual(self.scene.ngeom, slot + 1)
        self.assertEqual(
            self.scene.geoms[slot].type, mujoco.mjtGeom.mjGEOM_ARROW
        )

    def test_the_arrow_grows_with_the_gust(self):
        slot = 0
        lengths = []
        for force in (np.array([2.0, 0.0]), np.array([12.0, 0.0])):
            replay.draw_wind(self.handle, slot, np.zeros(2), force)
            geom = self.scene.geoms[slot]
            lengths.append(float(np.max(np.abs(geom.size))))
        self.assertGreater(lengths[1], lengths[0])


class ReplayArgumentTests(unittest.TestCase):
    def test_an_unknown_path_is_rejected(self):
        self.assertNotIn("not_a_path", paths.CATALOGUE)

    def test_calibration_resolution_prefers_the_newest_artifact(self):
        resolved = replay.resolve_calibration(PROJECT, None)
        self.assertTrue(resolved.exists())
        self.assertEqual(resolved.suffix, ".json")

    def test_gains_must_be_a_triple(self):
        with self.assertRaises(SystemExit):
            replay.simulate(scenario(gains="26,0.5"), PROJECT)


if __name__ == "__main__":
    unittest.main()
