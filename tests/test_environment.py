from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
from stable_baselines3.common.env_checker import check_env

from calibration import GainCalibration
import config
from core import paths
from core.dynamics import allocate_steering_priority, car_id, track_error
from env import FRAME_FIELDS, PathFollowingEnv, tracking_cost_of
from rollout import fixed_policy, run_episode


class GainMappingTests(unittest.TestCase):
    def setUp(self):
        self.calibration = GainCalibration(
            base=np.array([4.0, 1.0, 2.0]),
            low=np.array([1.0, 0.0, 0.5]),
            high=np.array([10.0, 3.0, 5.0]),
            metadata={},
        )

    def test_piecewise_mapping_hits_low_base_and_high(self):
        np.testing.assert_allclose(
            self.calibration.gains_from_action(np.full(3, -1.0)), self.calibration.low
        )
        np.testing.assert_allclose(
            self.calibration.gains_from_action(np.zeros(3)), self.calibration.base
        )
        np.testing.assert_allclose(
            self.calibration.gains_from_action(np.ones(3)), self.calibration.high
        )


class PlantContractTests(unittest.TestCase):
    def test_allocator_preserves_steering_and_bounds_wheels(self):
        _, _, left, right, v_applied, omega_applied = allocate_steering_priority(0.8, 0.6)
        self.assertAlmostEqual(v_applied, 0.4)
        self.assertAlmostEqual(omega_applied, 0.6)
        self.assertLessEqual(abs(left), 1.0)
        self.assertLessEqual(abs(right), 1.0)

    def test_reference_index_never_moves_backward(self):
        path = paths.get("figure8")
        first = track_error(path, path["pts"][100], 0.0, 100)
        second = track_error(path, path["pts"][10], 0.0, first.index)
        self.assertGreaterEqual(second.index, first.index)


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = PathFollowingEnv(training=False, path_keys=("arc",))

    def tearDown(self):
        self.env.close()

    def test_sb3_environment_contract(self):
        check_env(self.env, warn=True)

    def test_reset_preserves_exact_spawn_and_history_shape(self):
        observation, info = self.env.reset(
            seed=7, options={"path_key": "arc", "v_target": 0.3, "stage": 2}
        )
        np.testing.assert_allclose(self.env.data.qpos[0:3], [0.0, 0.0, 0.03])
        np.testing.assert_allclose(self.env.data.qpos[3:7], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(
            observation.shape, (len(FRAME_FIELDS) * config.HISTORY_LENGTH,)
        )
        self.assertEqual(info["path_key"], "arc")
        for privileged in ("mass", "friction", "actuator", "path_key", "progress", "distance"):
            self.assertNotIn(privileged, FRAME_FIELDS)

    def test_step_reports_bounded_actuators_and_gains(self):
        self.env.reset(
            seed=7, options={"path_key": "arc", "v_target": 0.3, "stage": 2}
        )
        observation, reward, terminated, truncated, info = self.env.step(
            np.ones(3, dtype=np.float32)
        )
        self.assertTrue(self.env.observation_space.contains(observation))
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertLessEqual(abs(info["u_left"]), 1.0 + 1e-12)
        self.assertLessEqual(abs(info["u_right"]), 1.0 + 1e-12)
        np.testing.assert_allclose(info["applied_gains"], self.env.calibration.high)

    def test_time_limit_is_a_truncation_and_failure(self):
        self.env.reset(
            seed=7,
            options={
                "path_key": "arc",
                "v_target": 0.3,
                "stage": 2,
                "max_episode_steps": 1,
            },
        )
        _, _, terminated, truncated, info = self.env.step(np.zeros(3))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["failure_reason"], "time_limit")
        self.assertLess(info["reward_terminal"], 0.0)

    def test_mass_disturbance_changes_plant_but_not_observation_schema(self):
        self.env.reset(
            seed=7,
            options={
                "path_key": "arc",
                "v_target": 0.3,
                "mass": 10.0,
                "stage": 3,
                "disturbance": {
                    "kind": "mass", "start_s": 0.0, "end_s": 1.0, "amount": 5.0
                },
            },
        )
        observation, _, _, _, _ = self.env.step(np.zeros(3))
        self.assertAlmostEqual(float(self.env.model.body_mass[car_id()]), 15.0)
        self.assertEqual(observation.shape, self.env.observation_space.shape)

    def test_persistent_mass_step_uses_an_absolute_target(self):
        self.env.reset(
            seed=7,
            options={
                "path_key": "arc",
                "v_target": 0.3,
                "mass": 10.0,
                "stage": 3,
                "disturbance": {
                    "kind": "mass_step", "start_s": 0.0, "end_s": None, "amount": 30.0
                },
            },
        )
        self.env.step(np.zeros(3))
        self.assertAlmostEqual(float(self.env.model.body_mass[car_id()]), 30.0)
        self.assertTrue(self.env.get_trace()[0]["disturbance_active"])

    def test_force_pulse_is_lateral_in_the_vehicle_frame(self):
        self.env.reset(
            seed=7,
            options={
                "path_key": "arc",
                "v_target": 0.3,
                "stage": 3,
                "disturbance": {
                    "kind": "force_pulse", "start_s": 0.0, "end_s": 1.0, "amount": 5.0
                },
            },
        )
        self.env.step(np.zeros(3))
        row = self.env.get_trace()[0]
        self.assertGreater(abs(row["external_force_y"]), 4.0)
        self.assertEqual(row["event_kind"], "force_pulse")

    def test_mass_and_force_disturbances_can_run_together(self):
        self.env.reset(
            seed=7,
            options={
                "path_key": "arc",
                "v_target": 0.3,
                "mass": 10.0,
                "stage": 3,
                "disturbances": [
                    {"kind": "mass_step", "start_s": 0.0, "end_s": None, "amount": 20.0},
                    {"kind": "force_pulse", "start_s": 0.0, "end_s": 1.0, "amount": 5.0},
                ],
            },
        )
        self.env.step(np.zeros(3))
        row = self.env.get_trace()[0]
        self.assertAlmostEqual(float(self.env.model.body_mass[car_id()]), 20.0)
        self.assertGreater(abs(row["external_force_y"]), 4.0)
        self.assertEqual(row["event_kind"], "mass_step+force_pulse")
        self.assertTrue(row["disturbance_active"])


class GustCurriculumTests(unittest.TestCase):
    """Stage 3 training must actually serve wind, and never the same wind twice."""

    def setUp(self):
        self.env = PathFollowingEnv(training=True, path_keys=("arc",))

    def tearDown(self):
        self.env.close()

    def _sample(self, episodes: int = 200) -> list[dict]:
        drawn = []
        for seed in range(episodes):
            self.env.reset(seed=seed, options={"stage": 3})
            drawn.append(self.env.disturbance)
        return drawn

    def test_stage_three_serves_all_three_event_kinds(self):
        kinds = {event["kind"] for event in self._sample()}
        self.assertEqual(kinds, {"mass", "force", "force_gust"})

    def test_gust_parameters_span_the_survivable_range(self):
        gusts = [e for e in self._sample() if e["kind"] == "force_gust"]
        self.assertGreater(len(gusts), 20)
        sigmas = [event["amount"] for event in gusts]
        low, high = config.GUST_SIGMA_RANGE_N
        self.assertGreaterEqual(min(sigmas), low)
        self.assertLessEqual(max(sigmas), high)
        # Wide coverage inside the survivable band matters: a mix clustered at
        # one strength teaches a single gain, not a rule. Training deliberately
        # stops below the regime where every controller fails regardless.
        self.assertLess(min(sigmas), 5.0)
        self.assertGreater(max(sigmas), 10.0)
        for event in gusts:
            self.assertGreaterEqual(event["tau_s"], config.GUST_TAU_RANGE_S[0])
            self.assertLessEqual(event["tau_s"], config.GUST_TAU_RANGE_S[1])

    def test_every_episode_draws_a_fresh_wind(self):
        gusts = [e for e in self._sample() if e["kind"] == "force_gust"]
        self.assertEqual(len({event["seed"] for event in gusts}), len(gusts))

    def test_the_same_episode_seed_reproduces_the_same_wind(self):
        self.env.reset(seed=11, options={"stage": 3})
        first = dict(self.env.disturbance)
        self.env.reset(seed=11, options={"stage": 3})
        self.assertEqual(first, dict(self.env.disturbance))

    def test_stage_two_stays_calm(self):
        self.env.reset(seed=3, options={"stage": 2})
        self.assertEqual(self.env.disturbance["kind"], "none")

    def test_stage_three_starts_before_half_of_training(self):
        """Stage 3 is the only stage serving disturbances; if it arrives late the
        agent barely meets them. Guards the value, not just the mechanism."""
        self.assertLessEqual(config.CURRICULUM_RANDOMIZED_END, 0.5)

    def test_gust_range_stops_short_of_the_unwinnable_regime(self):
        """Above ~12 N every controller leaves the corridor, so those episodes
        punish the agent whatever it does. 15 N stays a held-out test."""
        self.assertLessEqual(config.GUST_SIGMA_RANGE_N[1], 12.0)


class PlantImperfectionTests(unittest.TestCase):
    """Dead time and sensor noise: the mechanism, not yet the consequence.

    These exist because on the clean plant raising Kp was monotonically better
    until the wheel clamp truncated the command, so the best fixed gain sat on
    the box edge and the Fixed PID baseline matched the policy exactly. See
    GainTradeoffTests for the assertion that the tradeoff is actually there.
    """

    BASE_OPTIONS = {
        "path_key": "arc",
        "v_target": 0.5,
        "mass": config.NOMINAL_MASS_KG,
        "friction": config.NOMINAL_FRICTION,
        "actuator": config.NOMINAL_ACTUATOR,
        "stage": 3,
        "disturbances": [{"kind": "none", "start_s": 0.0, "end_s": None, "amount": 0.0}],
    }

    def _env(self, **kwargs) -> PathFollowingEnv:
        return PathFollowingEnv(training=False, path_keys=("arc",), **kwargs)

    def _reset(self, env: PathFollowingEnv, seed: int = 7, **overrides) -> None:
        env.reset(seed=seed, options={**self.BASE_OPTIONS, **overrides})

    def test_dead_time_withholds_the_command_for_the_configured_time(self):
        """One whole control step of delay means the wheels get nothing yet.

        The steering error is zero at spawn, so the request under test is the
        SPEED PID's: it asks for drive immediately, and with 0.02 s of dead time
        none of it may reach the actuators during the first control step.
        """
        env = self._env()
        try:
            self._reset(env, actuator_delay_s=config.FRAME_SKIP * 0.002)
            self.assertEqual(env.delay_steps, config.FRAME_SKIP)
            _, _, _, _, info = env.step(np.zeros(3))
            self.assertGreater(abs(info["u_left"]), 0.1)
            self.assertEqual(info["u_left_applied"], 0.0)
            self.assertEqual(info["u_right_applied"], 0.0)
            delayed_speed = abs(float(env.data.qvel[0]))

            # Against the same step with no dead time. An absolute threshold
            # would be wrong here: the car is still settling onto the floor
            # from its 0.03 m spawn height, so qvel is never exactly zero.
            self._reset(env, actuator_delay_s=0.0)
            env.step(np.zeros(3))
            self.assertGreater(abs(float(env.data.qvel[0])), 100.0 * delayed_speed)
        finally:
            env.close()

    def test_zero_delay_applies_the_command_in_the_same_substep(self):
        env = self._env()
        try:
            self._reset(env, actuator_delay_s=0.0)
            self.assertEqual(env.delay_steps, 0)
            _, _, _, _, info = env.step(np.zeros(3))
            self.assertEqual(info["u_left_applied"], info["u_left"])
            self.assertEqual(info["u_right_applied"], info["u_right"])
        finally:
            env.close()

    def test_delay_is_quantised_to_whole_physics_steps(self):
        """The command buffer has no sub-step resolution, so the stored seconds
        value must be the rounded one or traces would misreport the plant."""
        env = self._env()
        try:
            self._reset(env, actuator_delay_s=0.0035)
            self.assertEqual(env.delay_steps, 2)
            self.assertAlmostEqual(env.actuator_delay_s, 0.004)
        finally:
            env.close()

    def test_noise_corrupts_the_measurement_and_never_the_score(self):
        """Two error signals, kept apart on purpose.

        `dist` and the reward come from the true state; only the PID and the
        observation see the corrupted one. Scoring the controller on its own
        noisy measurement would report hallucinated performance.
        """
        env = self._env()
        try:
            self._reset(env, sensor_noise_m=0.002)
            for _ in range(5):
                env.step(np.zeros(3))
            rows = env.get_trace()
            residuals = [row["e_ct_measured"] - row["e_ct"] for row in rows]
            self.assertTrue(all(abs(r) > 0.0 for r in residuals))
            self.assertLess(max(abs(r) for r in residuals), 0.002 * 6.0)
            for row in rows:
                self.assertAlmostEqual(
                    row["dist"], float(np.hypot(row["e_ct"], 0.0)), delta=1.0
                )
        finally:
            env.close()

    def test_a_perfect_sensor_reports_the_truth_exactly(self):
        env = self._env()
        try:
            self._reset(env, sensor_noise_m=0.0)
            for _ in range(5):
                env.step(np.zeros(3))
            for row in env.get_trace():
                self.assertEqual(row["e_ct_measured"], row["e_ct"])
        finally:
            env.close()

    def test_both_controllers_meet_the_same_sensor_trace(self):
        """The noise stream is seeded independently of the episode generator.

        Same reason the gusts are: if the sensor trace were drawn from
        self.np_random it would depend on how many other draws each run
        happened to make, so the Fixed PID and the policy would be fighting
        different sensors and the paired comparison would mean nothing.
        """
        env = self._env()
        try:
            self._reset(env, seed=7, sensor_noise_m=0.002, noise_seed=5)
            for _ in range(5):
                env.step(np.zeros(3))
            first = [row["e_ct_measured"] for row in env.get_trace()]

            self._reset(env, seed=99, sensor_noise_m=0.002, noise_seed=5)
            for _ in range(5):
                env.step(np.zeros(3))
            second = [row["e_ct_measured"] for row in env.get_trace()]
        finally:
            env.close()
        self.assertEqual(first, second)

    def test_evaluation_episodes_are_clean_unless_asked(self):
        """Non-training resets take these from options, never the curriculum,
        so every existing evaluation stays reproducible until it opts in."""
        env = self._env()
        try:
            env.reset(seed=7, options={"path_key": "arc", "v_target": 0.5, "stage": 3})
            self.assertEqual(env.actuator_delay_s, 0.0)
            self.assertEqual(env.sensor_noise_m, 0.0)
        finally:
            env.close()

    def test_stage_one_is_a_perfect_plant_and_stage_three_is_not(self):
        env = PathFollowingEnv(training=True, path_keys=("arc",))
        try:
            env.reset(seed=3, options={"stage": 1})
            self.assertEqual(env.actuator_delay_s, 0.0)
            self.assertEqual(env.sensor_noise_m, 0.0)
            delays, noises = [], []
            for seed in range(200):
                env.reset(seed=seed, options={"stage": 3})
                delays.append(env.actuator_delay_s)
                noises.append(env.sensor_noise_m)
        finally:
            env.close()
        low_delay, high_delay = config.ACTUATOR_DELAY_RANGE_S
        low_noise, high_noise = config.SENSOR_NOISE_RANGE_M
        self.assertGreaterEqual(min(delays), low_delay)
        self.assertLessEqual(max(delays), high_delay)
        self.assertGreaterEqual(min(noises), low_noise)
        self.assertLessEqual(max(noises), high_noise)
        # Both extremes have to appear, for the same reason the gust range has
        # to straddle the point where the right answer inverts: a mix clustered
        # at one delay teaches a single gain rather than a rule.
        self.assertLess(min(delays), 0.25 * high_delay)
        self.assertGreater(max(delays), 0.75 * high_delay)
        self.assertGreater(max(noises), 0.75 * high_noise)

    def test_each_ablation_flag_removes_only_its_own_imperfection(self):
        """--no-delay and --no-noise must not shift the other's draw sequence.

        Both values are drawn unconditionally and then discarded, so an
        ablation run differs in the plant rather than in which random numbers
        it consumed. Without that, the control run is confounded.
        """
        full = PathFollowingEnv(training=True, path_keys=("arc",))
        no_delay = PathFollowingEnv(training=True, path_keys=("arc",), delay=False)
        no_noise = PathFollowingEnv(training=True, path_keys=("arc",), noise=False)
        try:
            for seed in range(25):
                full.reset(seed=seed, options={"stage": 3})
                no_delay.reset(seed=seed, options={"stage": 3})
                no_noise.reset(seed=seed, options={"stage": 3})
                self.assertEqual(no_delay.actuator_delay_s, 0.0)
                self.assertEqual(no_delay.sensor_noise_m, full.sensor_noise_m)
                self.assertEqual(no_noise.sensor_noise_m, 0.0)
                self.assertEqual(no_noise.actuator_delay_s, full.actuator_delay_s)
        finally:
            full.close()
            no_delay.close()
            no_noise.close()


class GainTradeoffTests(unittest.TestCase):
    """The precondition for the whole study: the best gain must MOVE.

    An interior optimum is not enough on its own. If dead time and noise merely
    lowered the best Kp from 50 to 30 everywhere, a Fixed PID re-tuned to 30
    would tie again and the adaptive policy would still have nothing to win.
    What makes gain scheduling necessary rather than merely possible is that
    the best gain differs by condition, so no single fixed choice is right
    everywhere. These tests assert that property directly, using the FIXED PID
    only -- no policy, no training. If they fail, no amount of PPO compute can
    produce a result and the plant needs fixing first.
    """

    CALIBRATION = Path(__file__).resolve().parents[1] / "runs" / "calibration_v5.json"

    @classmethod
    def setUpClass(cls):
        cls.calibration = GainCalibration.load(cls.CALIBRATION)
        cls.floor = (float(cls.calibration.low[0]), float(cls.calibration.low[2]))
        cls.ceiling = (float(cls.calibration.high[0]), float(cls.calibration.high[2]))

    def _error_cm(self, kp, kd, delay_s, noise_m, path_key, speed):
        """Mean tracking error in cm, or None if the episode did not finish."""
        gains = self.calibration.base.copy()
        gains[0], gains[2] = kp, kd
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
                    "actuator_delay_s": delay_s,
                    "sensor_noise_m": noise_m,
                    "noise_seed": config.SEED,
                    "stage": 3,
                    "disturbances": [
                        {"kind": "none", "start_s": 0.0, "end_s": None, "amount": 0.0}
                    ],
                },
            )
        finally:
            env.close()
        return metrics["mean_distance_m"] * 100 if metrics["finished"] else None

    def test_the_clean_plant_still_makes_high_gain_free(self):
        """Guards the DIAGNOSIS, not the fix.

        With no dead time and a perfect sensor, the box ceiling beats the box
        floor -- that is the behaviour that made PPO and the Fixed PID produce
        identical trajectories. If this test ever fails, something else changed
        the plant and the reasoning behind the ranges below no longer applies.
        """
        floor = self._error_cm(*self.floor, 0.0, 0.0, "zigzag", 0.7)
        ceiling = self._error_cm(*self.ceiling, 0.0, 0.0, "zigzag", 0.7)
        self.assertIsNotNone(floor)
        self.assertIsNotNone(ceiling)
        self.assertLess(ceiling, floor)

    def test_dead_time_and_noise_invert_the_ranking(self):
        """Same gains, same path, imperfect plant: the ceiling now LOSES.

        Measured 0.351 cm at the floor against 0.615 cm at the ceiling, a 1.75x
        penalty for the aggression that was free a moment ago.
        """
        delay_s = config.ACTUATOR_DELAY_RANGE_S[1]
        noise_m = config.SENSOR_NOISE_RANGE_M[1]
        floor = self._error_cm(*self.floor, delay_s, noise_m, "arc", 0.3)
        ceiling = self._error_cm(*self.ceiling, delay_s, noise_m, "arc", 0.3)
        self.assertIsNotNone(floor)
        self.assertIsNotNone(ceiling)
        self.assertLess(floor, ceiling)
        self.assertGreater(ceiling / floor, 1.3)

    def test_no_single_gain_choice_wins_on_every_condition(self):
        """The oracle gap, measured at the checkpoint-selection operating point.

        At 60 ms / 0.30 mm the best corner of the box is (Kp low, Kd low) on
        arc at 0.3 m/s and (Kp high, Kd high) on zigzag at 0.7 m/s -- opposite
        corners. Each is clearly worse than the other's winner on the other
        condition (1.4x and 1.8x), so the best achievable FIXED gain is beaten
        on both. That margin is the ceiling on what any adaptive policy can win
        here; if it collapses to nothing, the experiment cannot report a result.
        """
        delay_s = config.EVAL_ACTUATOR_DELAY_S
        noise_m = config.EVAL_SENSOR_NOISE_M
        corners = [
            (self.floor[0], self.floor[1]),
            (self.floor[0], self.ceiling[1]),
            (self.ceiling[0], self.floor[1]),
            (self.ceiling[0], self.ceiling[1]),
        ]
        winners = {}
        for path_key, speed in (("arc", 0.3), ("zigzag", 0.7)):
            scores = {}
            for kp, kd in corners:
                value = self._error_cm(kp, kd, delay_s, noise_m, path_key, speed)
                if value is not None:
                    scores[(kp, kd)] = value
            self.assertTrue(scores, f"no gain finished {path_key} @ {speed}")
            winners[(path_key, speed)] = (min(scores, key=scores.get), scores)

        (arc_best, arc_scores) = winners[("arc", 0.3)]
        (zig_best, zig_scores) = winners[("zigzag", 0.7)]
        self.assertNotEqual(
            arc_best, zig_best, "one gain pair won both conditions: nothing to schedule"
        )
        # Each condition's winner must be MEANINGFULLY worse on the other, or
        # the difference is a tie inside measurement scatter rather than a
        # tradeoff a policy could exploit.
        self.assertGreater(arc_scores[zig_best] / arc_scores[arc_best], 1.2)
        self.assertGreater(zig_scores[arc_best] / zig_scores[zig_best], 1.2)


class TrackingCostTests(unittest.TestCase):
    """The cost must resolve good controllers AND stay bounded for bad ones.

    A plain square at a 0.05 m scale satisfied the first and catastrophically
    failed the second: episode returns reached -475,734 against a median of
    +42, because the corridor tolerates 1.0 m and the time limit runs to ~2000
    steps. Both halves are asserted here so neither can regress alone.
    """

    def test_cost_resolves_sub_centimetre_differences(self):
        good = tracking_cost_of([0.0033] * 10)
        worse = tracking_cost_of([0.0045] * 10)
        self.assertGreater(worse / good, 1.2)

    def test_cost_saturates_instead_of_exploding(self):
        at_corridor = tracking_cost_of([1.0] * 10)
        far_beyond = tracking_cost_of([10.0] * 10)
        self.assertLess(at_corridor, 1.0)
        self.assertAlmostEqual(at_corridor, far_beyond, places=3)

    def test_worst_case_episode_stays_near_the_failure_penalty(self):
        """Bounded by construction: cost <= 1 per step, so the worst an episode
        can accrue is weight x steps. If that ever dwarfs the terminal terms,
        the value function cannot fit the return distribution."""
        longest_episode_steps = 2500
        worst = config.TRACKING_WEIGHT * 1.0 * longest_episode_steps
        self.assertLess(worst, 5 * (config.FINISH_BONUS + config.FAILURE_PENALTY))

    def test_cost_is_monotonic_in_distance(self):
        costs = [tracking_cost_of([d]) for d in (0.001, 0.005, 0.02, 0.1, 0.5)]
        self.assertEqual(costs, sorted(costs))


class RewardDiscriminationTests(unittest.TestCase):
    """The reward has to separate good gains from bad ones by a findable margin.

    It once did not: normalising tracking error by the corridor half-width made
    a 26% tracking improvement worth 0.03% of episode reward, so no amount of
    curriculum or training budget could have produced adaptation.
    """

    def _episode_reward(self, kp: float) -> tuple[float, float]:
        calibration = GainCalibration.load(
            Path(__file__).resolve().parents[1] / "runs" / "calibration_twosided.json"
        )
        gains = calibration.base.copy()
        gains[0] = kp
        env = PathFollowingEnv(
            calibration=calibration,
            training=False,
            path_keys=("arc",),
            fixed_gains=gains,
        )
        try:
            metrics, trace = run_episode(
                env,
                fixed_policy,
                seed=config.SEED,
                options={
                    "path_key": "arc",
                    "v_target": 0.5,
                    "mass": config.NOMINAL_MASS_KG,
                    "friction": config.NOMINAL_FRICTION,
                    "actuator": config.NOMINAL_ACTUATOR,
                    "stage": 3,
                    "disturbances": [
                        {
                            "kind": "force_gust",
                            "start_s": 0.0,
                            "end_s": None,
                            "amount": config.EVAL_GUST_SIGMA_N,
                            "tau_s": config.EVAL_GUST_TAU_S,
                            "seed": config.SEED,
                        }
                    ],
                },
            )
        finally:
            env.close()
        return sum(row["reward"] for row in trace), metrics["mean_distance_m"]

    def test_better_tracking_earns_meaningfully_more_reward(self):
        weak_reward, weak_distance = self._episode_reward(26.0)
        strong_reward, strong_distance = self._episode_reward(50.0)
        self.assertLess(strong_distance, weak_distance)
        self.assertGreater(strong_reward, weak_reward)
        margin = (strong_reward - weak_reward) / abs(weak_reward)
        # 5% is the floor for "PPO can find this through gradient noise". The
        # configured scale currently yields about 17%.
        self.assertGreater(margin, 0.05)

    def test_completion_still_outweighs_any_tracking_gain(self):
        """Guards the other direction: if tracking grew to dominate, the agent
        could prefer a tidy failure to a scruffy finish."""
        best, _ = self._episode_reward(50.0)
        worst, _ = self._episode_reward(26.0)
        self.assertLess(best - worst, config.FINISH_BONUS + config.FAILURE_PENALTY)


class EvaluationSuiteTests(unittest.TestCase):
    """The selection metric has to respond to the gains, or it selects noise."""

    def test_eval_gust_leaves_the_hardest_episode_winnable(self):
        """zigzag at 0.7 m/s is the hardest condition in the callback suite. If
        the eval gust sinks it, that one episode supplies most of the score and
        is insensitive to every gain the policy can choose."""
        calibration = GainCalibration.load(
            Path(__file__).resolve().parents[1] / "runs" / "calibration_twosided.json"
        )
        env = PathFollowingEnv(
            calibration=calibration,
            training=False,
            path_keys=("zigzag",),
            fixed_gains=calibration.base,
        )
        try:
            metrics, _ = run_episode(
                env,
                fixed_policy,
                seed=config.SEED,
                options={
                    "path_key": "zigzag",
                    "v_target": 0.7,
                    "mass": config.NOMINAL_MASS_KG,
                    "friction": config.NOMINAL_FRICTION,
                    "actuator": config.NOMINAL_ACTUATOR,
                    "stage": 3,
                    "disturbances": [
                        {
                            "kind": "force_gust",
                            "start_s": 0.0,
                            "end_s": None,
                            "amount": config.EVAL_GUST_SIGMA_N,
                            "tau_s": config.EVAL_GUST_TAU_S,
                            "seed": config.SEED,
                        }
                    ],
                },
            )
        finally:
            env.close()
        self.assertTrue(metrics["finished"])


class GustControlRunTests(unittest.TestCase):
    """--no-gusts must remove wind and change nothing else."""

    def test_control_run_serves_only_mass_and_force(self):
        env = PathFollowingEnv(training=True, path_keys=("arc",), gusts=False)
        try:
            kinds = set()
            for seed in range(200):
                env.reset(seed=seed, options={"stage": 3})
                kinds.add(env.disturbance["kind"])
        finally:
            env.close()
        self.assertEqual(kinds, {"mass", "force"})


if __name__ == "__main__":
    unittest.main()
