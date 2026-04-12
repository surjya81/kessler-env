"""
test_environment.py — tests for core KesslerEnvironment behaviour.

Covers: reset, gravity/physics, collision detection, task configuration,
reward shaping, episode termination, and score boundary guarantees.
"""
import math
import pytest # pyright: ignore[reportMissingImports]

from kessler_env_environment import ( # pyright: ignore[reportMissingImports]
    KesslerEnvironment,
    NUM_SATELLITES,
    MAX_STEPS,
    COLLISION_DIST,
    EARTH_RADIUS,
    _SCORE_EPSILON,
)
from models import KesslerAction, ThrusterBurn


# ===========================================================================
# Reset
# ===========================================================================

class TestReset:
    def test_satellites_spawned(self, env):
        assert len(env.satellites) == NUM_SATELLITES

    def test_all_satellites_active_after_reset(self, env):
        statuses = [s["status"] for s in env.satellites]
        assert all(s == "active" for s in statuses)

    def test_step_count_zero_after_reset(self, env):
        assert env._state.step_count == 0

    def test_task1_debris_count(self, env):
        # Task 1: 25 debris pieces
        assert len(env.debris) == 25

    def test_task2_debris_count(self, env_task2):
        # Task 2: 35 debris pieces
        assert len(env_task2.debris) == 35

    def test_task3_debris_count(self, env_task3):
        # Task 3: 50 debris pieces (plus potential rogue spawns during steps)
        assert len(env_task3.debris) == 50

    def test_task1_fuel(self, env):
        fuels = [s["fuel"] for s in env.satellites]
        assert all(f == 100.0 for f in fuels)

    def test_task2_fuel(self, env_task2):
        fuels = [s["fuel"] for s in env_task2.satellites]
        assert all(f == 50.0 for f in fuels)

    def test_task3_has_target_radius(self, env_task3):
        assert env_task3.target_radius == 100.0

    def test_task1_no_target_radius(self, env):
        assert env.target_radius == 0.0

    def test_satellites_not_inside_earth(self, env):
        for s in env.satellites:
            r = math.sqrt(s["x"] ** 2 + s["y"] ** 2)
            assert r > EARTH_RADIUS, f"Sat {s['id']} spawned inside Earth: r={r:.2f}"

    def test_episode_id_changes_on_reset(self, env):
        old_id = env._state.episode_id
        env.reset()
        assert env._state.episode_id != old_id


# ===========================================================================
# Step — basic mechanics
# ===========================================================================

class TestStep:
    def test_step_increments_count(self, env, no_burn):
        env.step(no_burn)
        assert env._state.step_count == 1

    def test_step_returns_observation(self, env, no_burn):
        obs = env.step(no_burn)
        assert obs is not None
        assert len(obs.satellites) == NUM_SATELLITES

    def test_reward_is_nonnegative(self, env, no_burn):
        obs = env.step(no_burn)
        assert obs.reward >= 0.0

    def test_done_false_before_max_steps(self, env, no_burn):
        obs = env.step(no_burn)
        assert obs.done is False

    def test_done_true_at_max_steps(self, env, no_burn):
        """Simulate MAX_STEPS steps and verify done triggers."""
        obs = None
        for _ in range(MAX_STEPS):
            obs = env.step(no_burn)
        assert obs.done is True

    def test_no_burn_costs_no_fuel(self, env, no_burn):
        initial_fuel = [s["fuel"] for s in env.satellites]
        env.step(no_burn)
        current_fuel = [s["fuel"] for s in env.satellites]
        assert initial_fuel == current_fuel

    def test_burn_costs_fuel(self, env):
        burn = KesslerAction(burns=[ThrusterBurn(satellite_id=0, delta_vx=1.0, delta_vy=0.0)])
        fuel_before = next(s["fuel"] for s in env.satellites if s["id"] == 0)
        env.step(burn)
        fuel_after = next(s["fuel"] for s in env.satellites if s["id"] == 0)
        assert fuel_after < fuel_before

    def test_burn_clamps_delta_v(self, env):
        """Delta-v beyond ±1.0 should be clamped, not rejected."""
        oversized_burn = KesslerAction(burns=[
            ThrusterBurn(satellite_id=0, delta_vx=99.0, delta_vy=-99.0)
        ])
        # Should not raise — clamped internally
        obs = env.step(oversized_burn)
        assert obs is not None

    def test_burn_on_nonexistent_satellite_is_ignored(self, env):
        bad_burn = KesslerAction(burns=[ThrusterBurn(satellite_id=99, delta_vx=1.0, delta_vy=0.0)])
        obs = env.step(bad_burn)
        # No crash, satellites unchanged
        assert len(obs.satellites) == NUM_SATELLITES

    def test_insufficient_fuel_generates_alert(self, env):
        """Drain a satellite's fuel then try to burn — should emit an alert."""
        sat = next(s for s in env.satellites if s["id"] == 0)
        sat["fuel"] = 0.001  # nearly empty
        burn = KesslerAction(burns=[ThrusterBurn(satellite_id=0, delta_vx=1.0, delta_vy=0.0)])
        obs = env.step(burn)
        assert any("fuel" in a.lower() for a in obs.critical_alerts)


# ===========================================================================
# Reward shaping
# ===========================================================================

class TestRewardShaping:
    def test_step_reward_positive_when_all_alive(self, env, no_burn):
        obs = env.step(no_burn)
        assert obs.reward > 0.0

    def test_step_reward_is_zero_when_all_destroyed(self, env, no_burn):
        """Manually destroy all satellites, then step."""
        for s in env.satellites:
            s["status"] = "destroyed"
        obs = env.step(no_burn)
        assert obs.reward == 0.0

    def test_reward_ceiling_never_hits_1(self, env, no_burn):
        """Sum of rewards over a perfect run must stay below 1.0."""
        total = 0.0
        for _ in range(MAX_STEPS):
            obs = env.step(no_burn)
            total += obs.reward
        assert total < 1.0, f"Reward sum reached {total} — ceiling violated"

    def test_score_epsilon_floor(self):
        """_SCORE_EPSILON is small enough not to overflow on its own."""
        assert 0 < _SCORE_EPSILON < 0.01

    def test_task2_eco_penalty_applied(self, env_task2):
        """Burning on ECO-STATION should reduce reward compared to no-burn."""
        # Measure reward with no burn
        import copy
        env_no_burn = copy.deepcopy(env_task2)
        obs_no_burn = env_no_burn.step(KesslerAction(burns=[]))

        # Measure reward with maximum burn
        env_burn = copy.deepcopy(env_task2)
        big_burn = KesslerAction(burns=[
            ThrusterBurn(satellite_id=i, delta_vx=1.0, delta_vy=0.0)
            for i in range(NUM_SATELLITES)
        ])
        obs_burn = env_burn.step(big_burn)

        # Burning should reduce reward on ECO-STATION
        assert obs_burn.reward < obs_no_burn.reward

    def test_task3_proximity_bonus_increases_reward(self, env_task3, no_burn):
        """Satellite 0 close to target_radius=100 should earn a higher reward."""
        import copy
        import math as _math
        
        # 1. Measure ON-TARGET using a deep copy
        env_on_target = copy.deepcopy(env_task3)
        sat0 = next(s for s in env_on_target.satellites if s["id"] == 0)
        r = 100.0
        sat0["x"] = r
        sat0["y"] = 0.0
        v = _math.sqrt(1000.0 / r)
        sat0["vx"] = 0.0
        sat0["vy"] = v

        obs_on_target = env_on_target.step(no_burn)

        # 2. Measure OFF-TARGET using a fresh deep copy
        # (This prevents advancing the episode count to Task 1)
        env_off_target = copy.deepcopy(env_task3)
        sat0_far = next(s for s in env_off_target.satellites if s["id"] == 0)
        sat0_far["x"] = 200.0   # far from target_radius=100
        sat0_far["y"] = 0.0
        
        obs_off_target = env_off_target.step(no_burn)

        # 3. Task 3 reward with proximity bonus should beat Task 3 reward without it
        assert obs_on_target.reward > obs_off_target.reward


# ===========================================================================
# Collision & Kessler cascade
# ===========================================================================

class TestCollision:
    def _place_debris_on_satellite(self, env, sat_id: int, debris_idx: int = 0):
        """
        Place debris at a satellite's exact position AND copy its velocity.
        Gravity runs before collision detection in step(), so if velocities
        differ the objects drift apart before the collision check runs.
        Matching velocity ensures both move identically under gravity.
        """
        sat = next(s for s in env.satellites if s["id"] == sat_id)
        env.debris[debris_idx]["x"]  = sat["x"]
        env.debris[debris_idx]["y"]  = sat["y"]
        env.debris[debris_idx]["vx"] = sat["vx"]
        env.debris[debris_idx]["vy"] = sat["vy"]

    def test_collision_destroys_satellite(self, env, no_burn):
        """Debris co-located with Sat 0 (same velocity) must destroy it."""
        self._place_debris_on_satellite(env, sat_id=0)
        env.step(no_burn)
        sat0_after = next(s for s in env.satellites if s["id"] == 0)
        assert sat0_after["status"] == "destroyed"

    def test_collision_spawns_cascade_fragments(self, env, no_burn):
        """A collision should spawn 2 new debris fragments."""
        debris_before = len(env.debris)
        self._place_debris_on_satellite(env, sat_id=0)
        env.step(no_burn)
        assert len(env.debris) == debris_before + 2

    def test_collision_emits_alert(self, env, no_burn):
        self._place_debris_on_satellite(env, sat_id=0)
        obs = env.step(no_burn)
        assert any("CRITICAL" in a for a in obs.critical_alerts)

    def test_all_destroyed_ends_episode(self, env, no_burn):
        """When all satellites are destroyed the episode should be done."""
        for s in env.satellites:
            s["status"] = "destroyed"
        obs = env.step(no_burn)
        assert obs.done is True

    def test_orbit_decay_destroys_satellite(self, env, no_burn):
        """Satellite inside EARTH_RADIUS should be marked destroyed."""
        sat0 = next(s for s in env.satellites if s["id"] == 0)
        sat0["x"] = EARTH_RADIUS * 0.5   # inside Earth
        sat0["y"] = 0.0
        env.step(no_burn)
        sat0_after = next(s for s in env.satellites if s["id"] == 0)
        assert sat0_after["status"] == "destroyed"