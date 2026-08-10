from __future__ import annotations

import unittest

import numpy as np
from stable_baselines3.common.env_checker import check_env

import config
from core import paths
from env import CONTEXT_FIELDS, FRAME_FIELDS, PathFollowingEnv


BASE_WIDTH = len(FRAME_FIELDS) * config.HISTORY_LENGTH
OPTIONS = {
    "path_key": "zigzag",
    "v_target": 0.5,
    "mass": 30.0,
    "friction": 0.5,
    "actuator": 1.0,
    "actuator_delay_s": 0.06,
    "sensor_noise_m": 0.0003,
    "noise_seed": 7,
    "stage": 3,
    "disturbances": [{"kind": "none", "start_s": 0.0, "end_s": None, "amount": 0.0}],
}


class CurvatureTests(unittest.TestCase):
    """Preview is only worth having if the curvature it reports is right."""

    def test_a_straight_line_has_no_curvature(self):
        path = paths.straight(6.0)
        np.testing.assert_allclose(paths.curvature_of(path), 0.0, atol=1e-9)

    def test_an_arc_reports_one_over_its_radius(self):
        path = paths.get("arc")  # 3 m radius, left-hand
        kappa = paths.curvature_of(path)
        middle = kappa[len(kappa) // 4 : 3 * len(kappa) // 4]
        self.assertAlmostEqual(float(np.median(middle)), 1.0 / 3.0, places=2)
        self.assertGreater(float(np.median(middle)), 0.0, "left turns are positive")

    def test_curvature_is_signed_and_reverses_on_an_s_curve(self):
        kappa = paths.curvature_of(paths.get("scurve"))
        self.assertGreater(kappa.max(), 0.0)
        self.assertLess(kappa.min(), 0.0)

    def test_preview_looks_ahead_and_holds_past_the_end(self):
        path = paths.get("arc")
        kappa = paths.curvature_of(path)
        ahead = paths.preview_at(path, kappa, 0.0, config.PREVIEW_DISTANCES_M)
        self.assertEqual(len(ahead), len(config.PREVIEW_DISTANCES_M))
        # Far past the finish, the last value is held rather than wrapping to
        # the start or dropping to zero -- a phantom straight at the end would
        # be worse than no preview.
        beyond = paths.preview_at(path, kappa, path["length"] + 50.0, (0.0, 10.0))
        np.testing.assert_allclose(beyond[0], beyond[1])


class ObservationWidthTests(unittest.TestCase):
    def test_each_arm_adds_its_own_block(self):
        cases = {
            (False, False): BASE_WIDTH,
            (True, False): BASE_WIDTH + len(config.PREVIEW_DISTANCES_M),
            (False, True): BASE_WIDTH + len(CONTEXT_FIELDS),
            (True, True): BASE_WIDTH + len(config.PREVIEW_DISTANCES_M) + len(CONTEXT_FIELDS),
        }
        for (preview, context), width in cases.items():
            env = PathFollowingEnv(
                training=False, path_keys=("arc",),
                preview=preview, plant_context=context,
            )
            try:
                self.assertEqual(env.observation_space.shape, (width,))
                observation, _ = env.reset(seed=7, options=dict(OPTIONS, path_key="arc"))
                self.assertEqual(observation.shape, (width,))
                self.assertTrue(env.observation_space.contains(observation))
            finally:
                env.close()

    def test_the_blind_default_is_unchanged(self):
        """Existing artifacts must stay loadable: both arms default off."""
        env = PathFollowingEnv(training=False, path_keys=("arc",))
        try:
            self.assertFalse(env.preview)
            self.assertFalse(env.plant_context)
            self.assertEqual(env.observation_space.shape, (BASE_WIDTH,))
        finally:
            env.close()

    def test_both_arms_satisfy_the_sb3_contract(self):
        env = PathFollowingEnv(
            training=False, path_keys=("arc",), preview=True, plant_context=True
        )
        try:
            check_env(env, warn=True)
        finally:
            env.close()


class PreviewContentTests(unittest.TestCase):
    def test_the_preview_advances_with_the_car(self):
        env = PathFollowingEnv(training=False, path_keys=("zigzag",), preview=True)
        try:
            env.reset(seed=7, options=OPTIONS)
            first = env._preview_block().copy()
            for _ in range(40):
                env.step(np.zeros(3))
            later = env._preview_block().copy()
        finally:
            env.close()
        self.assertFalse(
            np.allclose(first, later),
            "preview is fixed at reset instead of tracking the car's position",
        )

    def test_preview_sees_a_corner_before_the_car_reaches_it(self):
        """The whole point: curvature ahead must be non-zero while the car is
        still on the straight leading into it."""
        env = PathFollowingEnv(training=False, path_keys=("square",), preview=True)
        try:
            env.reset(seed=7, options=dict(OPTIONS, path_key="square"))
            saw_corner_ahead_while_straight = False
            for _ in range(200):
                here = float(env._path_curvature[env.path_index])
                ahead = env._preview_block()
                if abs(here) < 1e-6 and np.abs(ahead).max() > 0.05:
                    saw_corner_ahead_while_straight = True
                    break
                _, _, terminated, truncated, _ = env.step(np.zeros(3))
                if terminated or truncated:
                    break
        finally:
            env.close()
        self.assertTrue(saw_corner_ahead_while_straight)


class PlantContextTests(unittest.TestCase):
    def test_the_context_block_reports_the_hidden_parameters(self):
        env = PathFollowingEnv(training=False, path_keys=("zigzag",), plant_context=True)
        try:
            env.reset(seed=7, options=OPTIONS)
            block = env._context_block()
            self.assertEqual(len(block), len(CONTEXT_FIELDS))
            self.assertAlmostEqual(
                float(block[0]), 30.0 / config.CONTEXT_MASS_SCALE_KG, places=5
            )
            self.assertAlmostEqual(
                float(block[1]), 0.5 / config.CONTEXT_FRICTION_SCALE, places=5
            )
            self.assertAlmostEqual(
                float(block[3]), 0.06 / config.CONTEXT_DELAY_SCALE_S, places=5
            )
        finally:
            env.close()

    def test_a_mid_episode_mass_step_is_visible_immediately(self):
        """Mass is read from the live model, not the episode's base value, so
        the teacher sees the step the instant it lands. That immediacy is the
        advantage the blind student has to infer from the error history."""
        env = PathFollowingEnv(training=False, path_keys=("arc",), plant_context=True)
        try:
            env.reset(seed=7, options=dict(
                OPTIONS, path_key="arc", mass=10.0,
                disturbances=[{"kind": "mass_step", "start_s": 0.5,
                               "end_s": None, "amount": 45.0}],
            ))
            before = float(env._context_block()[0])
            while env.time_s < 0.8:
                _, _, terminated, truncated, _ = env.step(np.zeros(3))
                if terminated or truncated:
                    break
            after = float(env._context_block()[0])
        finally:
            env.close()
        self.assertGreater(after, before * 2.0)

    def test_the_blind_agent_cannot_see_any_of_it(self):
        env = PathFollowingEnv(training=False, path_keys=("zigzag",))
        try:
            observation, _ = env.reset(seed=7, options=OPTIONS)
            self.assertEqual(len(observation), BASE_WIDTH)
        finally:
            env.close()
        for privileged in ("mass", "friction", "actuator", "actuator_delay_s",
                           "sensor_noise_m"):
            self.assertNotIn(privileged, FRAME_FIELDS)


if __name__ == "__main__":
    unittest.main()
