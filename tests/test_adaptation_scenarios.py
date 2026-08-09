from __future__ import annotations

import unittest

from adaptation_scenarios import (
    EventDefinition,
    PhysicsConfig,
    Scenario,
    build_scenarios,
    smoke_scenarios,
    validate_scenario_set,
)


class ScenarioContractTests(unittest.TestCase):
    def test_roundtrip_preserves_fingerprint_and_event_timing(self):
        scenario = Scenario(
            "slalom_v0p5_mass_10_to_30",
            "slalom",
            0.5,
            7,
            PhysicsConfig(mass=10.0),
            EventDefinition("mass_step", 4.25, None, 30.0),
            ("development", "dynamic", "mass"),
        )
        restored = Scenario.from_dict(scenario.to_dict())
        self.assertEqual(restored, scenario)
        self.assertEqual(restored.fingerprint, scenario.fingerprint)
        options = restored.to_env_options()
        self.assertEqual(options["disturbance"]["start_s"], 4.25)
        self.assertEqual(options["disturbance"]["amount"], 30.0)

    def test_invalid_ranges_and_event_windows_are_rejected(self):
        with self.assertRaises(ValueError):
            PhysicsConfig(mass=100.0)
        with self.assertRaises(ValueError):
            EventDefinition("force_pulse", 3.0, 2.0, 5.0)
        with self.assertRaises(ValueError):
            EventDefinition("mass_step", 3.0, 4.0, 30.0)

    def test_scenario_ids_are_unique_and_smoke_covers_event_families(self):
        full = build_scenarios(("arc",), (0.3,))
        validate_scenario_set(full)
        self.assertEqual(len({item.scenario_id for item in full}), len(full))
        smoke = smoke_scenarios()
        self.assertEqual(len(smoke), 4)
        self.assertEqual({item.event.kind for item in smoke}, {"none", "mass_step", "force_pulse"})


if __name__ == "__main__":
    unittest.main()
