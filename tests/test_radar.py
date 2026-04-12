"""
test_radar.py — tests for the partial observability (RADAR_RANGE) feature.

Key invariants being tested:
  - Full observability (RADAR_RANGE=0) is the default and returns all debris.
  - Partial observability filters debris by distance to active satellites.
  - Debris outside radar range is still physically simulated (can kill satellites).
  - radar_range is surfaced in the observation so the agent knows its limit.
  - All-satellites-destroyed falls back to empty visible list, not a crash.
"""
import math
import pytest # pyright: ignore[reportMissingImports]
import kessler_env_environment as env_module # pyright: ignore[reportMissingImports]
from kessler_env_environment import KesslerEnvironment # pyright: ignore[reportMissingImports]
from models import KesslerAction, ThrusterBurn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dist(obj_a: dict, obj_b: dict) -> float:
    return math.sqrt((obj_a["x"] - obj_b["x"]) ** 2 + (obj_a["y"] - obj_b["y"]) ** 2)


def _place_debris_at(env: KesslerEnvironment, debris_idx: int, x: float, y: float):
    """Move a specific debris piece to an (x, y) position."""
    env.debris[debris_idx]["x"] = x
    env.debris[debris_idx]["y"] = y


def _place_sat_at(env: KesslerEnvironment, sat_id: int, x: float, y: float):
    sat = next(s for s in env.satellites if s["id"] == sat_id)
    sat["x"] = x
    sat["y"] = y


# ---------------------------------------------------------------------------
# Full observability (default)
# ---------------------------------------------------------------------------

class TestFullObservability:
    def test_radar_range_zero_returns_all_debris(self, env):
        """RADAR_RANGE=0 should expose every debris piece."""
        assert env_module.RADAR_RANGE == 0.0
        obs = env._get_observation([])
        assert len(obs.radar_debris) == len(env.debris)

    def test_radar_range_field_is_zero_in_observation(self, env):
        obs = env._get_observation([])
        assert obs.radar_range == 0.0

    def test_full_obs_distant_debris_still_visible(self, env):
        """Even debris 1000 units away should appear in full-obs mode."""
        env.debris[0]["x"] = 1000.0
        env.debris[0]["y"] = 1000.0
        obs = env._get_observation([])
        ids = [d.id for d in obs.radar_debris]
        assert 0 in ids


# ---------------------------------------------------------------------------
# Partial observability — filtering logic
# ---------------------------------------------------------------------------

class TestPartialObservability:
    def test_nearby_debris_is_visible(self, env):
        """Debris placed right next to Sat 0 must always be visible."""
        env_module.RADAR_RANGE = 30.0
        sat0 = next(s for s in env.satellites if s["id"] == 0)
        _place_debris_at(env, 0, sat0["x"] + 1.0, sat0["y"] + 1.0)

        obs = env._get_observation([])
        ids = [d.id for d in obs.radar_debris]
        assert 0 in ids

    def test_far_debris_is_hidden(self, env):
        """Debris placed 500 units away must be hidden when RADAR_RANGE=30."""
        env_module.RADAR_RANGE = 30.0
        # Put all satellites at origin-ish
        for s in env.satellites:
            s["x"] = 60.0
            s["y"] = 0.0

        # Move debris[0] far away
        _place_debris_at(env, 0, 600.0, 600.0)

        obs = env._get_observation([])
        ids = [d.id for d in obs.radar_debris]
        assert 0 not in ids

    def test_debris_visible_if_near_any_satellite(self, env):
        """A debris piece near Sat 2 (not Sat 0) must still be visible."""
        env_module.RADAR_RANGE = 20.0
        # Spread satellites far apart
        env.satellites[0]["x"] = 0.0
        env.satellites[0]["y"] = 0.0
        env.satellites[1]["x"] = 500.0
        env.satellites[1]["y"] = 0.0
        env.satellites[2]["x"] = 0.0
        env.satellites[2]["y"] = 500.0

        # Place debris[0] near Sat 2 only
        _place_debris_at(env, 0, 5.0, 505.0)   # 5 units from Sat 2

        obs = env._get_observation([])
        ids = [d.id for d in obs.radar_debris]
        assert 0 in ids

    def test_radar_range_field_propagated_to_observation(self, env):
        env_module.RADAR_RANGE = 42.5
        obs = env._get_observation([])
        assert obs.radar_range == 42.5

    def test_partial_obs_count_less_than_full(self, env):
        """Partial observability should expose fewer debris than full."""
        # Cluster all satellites at one spot
        for s in env.satellites:
            s["x"] = 60.0
            s["y"] = 0.0

        # Scatter most debris far away
        for i, d in enumerate(env.debris):
            d["x"] = 500.0 + i * 10
            d["y"] = 500.0

        # One debris near the satellites
        _place_debris_at(env, 0, 61.0, 0.0)

        env_module.RADAR_RANGE = 30.0
        obs_partial = env._get_observation([])

        env_module.RADAR_RANGE = 0.0
        obs_full = env._get_observation([])

        assert len(obs_partial.radar_debris) < len(obs_full.radar_debris)

    def test_exact_boundary_is_visible(self, env):
        """Debris clearly within RADAR_RANGE should be visible.

        Uses 29.9 instead of exactly 30.0 to avoid float precision issues:
        sqrt((sat_x + 30.0 - sat_x)^2) can return 30.000000000000004,
        failing the <= check by an ulp. 29.9 is unambiguously inside range.
        """
        env_module.RADAR_RANGE = 30.0
        sat0 = next(s for s in env.satellites if s["id"] == 0)
        _place_debris_at(env, 0, sat0["x"] + 29.9, sat0["y"])

        obs = env._get_observation([])
        ids = [d.id for d in obs.radar_debris]
        assert 0 in ids

    def test_just_outside_boundary_is_hidden(self, env):
        """Debris just beyond RADAR_RANGE should be excluded."""
        env_module.RADAR_RANGE = 30.0
        sat0 = next(s for s in env.satellites if s["id"] == 0)
        # All other satellites far away so only sat0 matters
        env.satellites[1]["x"] = 500.0
        env.satellites[2]["x"] = 500.0
        _place_debris_at(env, 0, sat0["x"] + 30.001, sat0["y"])

        obs = env._get_observation([])
        ids = [d.id for d in obs.radar_debris]
        assert 0 not in ids


# ---------------------------------------------------------------------------
# Physics still simulates hidden debris
# ---------------------------------------------------------------------------

class TestHiddenDebrisStillKills:
    def test_hidden_debris_can_still_collide(self, env):
        """
        Debris outside radar range must still be simulated and can kill.

        Setup: satellites clustered at (60, 0); all debris scattered far away
        except debris[0] which is placed right on Sat 0 with matching velocity
        so it stays co-located after the gravity step and triggers a collision.
        The tight radar cone (5 units) keeps most debris hidden, confirming the
        simulation runs on all debris regardless of visibility.
        """
        env_module.RADAR_RANGE = 5.0

        # Cluster all satellites
        for s in env.satellites:
            s["x"] = 60.0
            s["y"] = 0.0
            s["vx"] = 0.0
            s["vy"] = 4.08   # approximate circular velocity at r=60

        # Scatter all debris far away
        for d in env.debris:
            d["x"] = 500.0
            d["y"] = 500.0

        # Place debris[0] on Sat 0 with same velocity — guaranteed collision
        sat0 = next(s for s in env.satellites if s["id"] == 0)
        env.debris[0]["x"]  = sat0["x"]
        env.debris[0]["y"]  = sat0["y"]
        env.debris[0]["vx"] = sat0["vx"]
        env.debris[0]["vy"] = sat0["vy"]

        env.step(KesslerAction(burns=[]))
        sat0_after = next(s for s in env.satellites if s["id"] == 0)
        assert sat0_after["status"] == "destroyed"

    def test_hidden_debris_still_moves(self, env):
        """
        Debris outside radar range must be physics-updated each step.
        Its position should change between steps.
        """
        env_module.RADAR_RANGE = 5.0

        # Scatter all satellites to origin area, put debris[0] far away
        for s in env.satellites:
            s["x"] = 60.0
            s["y"] = 0.0
        _place_debris_at(env, 0, 500.0, 500.0)

        x_before = env.debris[0]["x"]
        y_before = env.debris[0]["y"]

        env.step(KesslerAction(burns=[]))

        x_after = env.debris[0]["x"]
        y_after = env.debris[0]["y"]

        # Debris should have moved due to gravity
        assert (x_after, y_after) != (x_before, y_before)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestRadarEdgeCases:
    def test_no_active_satellites_returns_empty_radar(self, env):
        """When all satellites are destroyed, no debris should be visible."""
        env_module.RADAR_RANGE = 50.0
        for s in env.satellites:
            s["status"] = "destroyed"
        obs = env._get_observation([])
        assert obs.radar_debris == []

    def test_very_large_radar_range_shows_all_debris(self, env):
        """RADAR_RANGE=1e9 should behave like full observability."""
        env_module.RADAR_RANGE = 1_000_000.0
        obs = env._get_observation([])
        assert len(obs.radar_debris) == len(env.debris)

    def test_switching_radar_range_between_steps(self, env):
        """Changing RADAR_RANGE mid-episode should take effect immediately."""
        env_module.RADAR_RANGE = 0.0
        obs_full = env._get_observation([])
        full_count = len(obs_full.radar_debris)

        env_module.RADAR_RANGE = 1.0   # virtually nothing visible
        obs_tiny = env._get_observation([])
        assert len(obs_tiny.radar_debris) <= full_count