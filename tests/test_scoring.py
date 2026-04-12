"""
test_scoring.py — validates that episode scores always land strictly
inside (0, 1), as required by the hackathon grader.

These tests replicate the exact scoring logic from inference.py so any
future change to either file will be caught immediately.
"""
import pytest # pyright: ignore[reportMissingImports]
from kessler_env_environment import ( # pyright: ignore[reportMissingImports]
    KesslerEnvironment,
    MAX_STEPS,
    NUM_SATELLITES,
    _SCORE_EPSILON,
) 
from models import KesslerAction, ThrusterBurn

# Mirror the inference.py scoring constants exactly
_INFERENCE_SCORE_EPSILON = 1e-3
_MAX_TOTAL_REWARD = 1.0


def _simulate_and_score(env: KesslerEnvironment, action: KesslerAction) -> float:
    """Run a full episode with a fixed action and return the final score."""
    rewards = []
    for _ in range(MAX_STEPS):
        obs = env.step(action)
        rewards.append(obs.reward)
        if obs.done:
            break

    raw = sum(rewards) / _MAX_TOTAL_REWARD
    score = raw + _INFERENCE_SCORE_EPSILON
    score = min(score, 1.0 - _INFERENCE_SCORE_EPSILON)
    return score


class TestScoreBounds:
    def test_perfect_run_score_below_one(self, env):
        """A perfect survival run (all satellites alive) must score < 1.0."""
        score = _simulate_and_score(env, KesslerAction(burns=[]))
        assert score < 1.0, f"Score hit ceiling: {score}"

    def test_perfect_run_score_above_zero(self, env):
        score = _simulate_and_score(env, KesslerAction(burns=[]))
        assert score > 0.0, f"Score hit floor: {score}"

    def test_catastrophic_run_score_above_zero(self, env):
        """Even if all satellites die on step 1, score must be > 0."""
        for s in env.satellites:
            s["status"] = "destroyed"
        score = _simulate_and_score(env, KesslerAction(burns=[]))
        assert score > 0.0, f"Score hit floor on total loss: {score}"

    def test_catastrophic_run_score_below_one(self, env):
        for s in env.satellites:
            s["status"] = "destroyed"
        score = _simulate_and_score(env, KesslerAction(burns=[]))
        assert score < 1.0

    def test_score_strictly_inside_open_interval(self, env):
        score = _simulate_and_score(env, KesslerAction(burns=[]))
        assert 0.0 < score < 1.0

    def test_score_epsilon_is_consistent(self):
        """_SCORE_EPSILON in environment and inference must match."""
        assert _SCORE_EPSILON == _INFERENCE_SCORE_EPSILON, (
            f"Epsilon mismatch: environment={_SCORE_EPSILON}, "
            f"inference={_INFERENCE_SCORE_EPSILON} — keep these in sync"
        )

    def test_reward_sum_never_exceeds_one(self, env):
        """Raw reward sum (before epsilon adjustment) must be < 1.0."""
        rewards = []
        for _ in range(MAX_STEPS):
            obs = env.step(KesslerAction(burns=[]))
            rewards.append(obs.reward)
            if obs.done:
                break
        assert sum(rewards) < 1.0, f"Raw reward sum = {sum(rewards):.6f} >= 1.0"

    def test_reward_sum_nonnegative(self, env):
        rewards = []
        for _ in range(MAX_STEPS):
            obs = env.step(KesslerAction(burns=[]))
            rewards.append(obs.reward)
        assert sum(rewards) >= 0.0

    def test_task2_eco_score_in_bounds(self, env_task2):
        score = _simulate_and_score(env_task2, KesslerAction(burns=[]))
        assert 0.0 < score < 1.0

    def test_task3_rendezvous_score_in_bounds(self, env_task3):
        score = _simulate_and_score(env_task3, KesslerAction(burns=[]))
        assert 0.0 < score < 1.0

    def test_all_three_tasks_in_bounds(self, env, env_task2, env_task3):
        """Batch check — all tasks must produce grader-valid scores."""
        for label, e in [("task1", env), ("task2", env_task2), ("task3", env_task3)]:
            score = _simulate_and_score(e, KesslerAction(burns=[]))
            assert 0.0 < score < 1.0, f"{label} score out of range: {score}"


class TestPerStepReward:
    def test_step_reward_nonnegative(self, env):
        obs = env.step(KesslerAction(burns=[]))
        assert obs.reward >= 0.0

    def test_step_reward_ceiling(self, env):
        """Single step reward must be < (1 - 2*epsilon) / MAX_STEPS ceiling."""
        ceiling = (1.0 - 2 * _SCORE_EPSILON) / MAX_STEPS
        obs = env.step(KesslerAction(burns=[]))
        # Task 3 can double the base reward with proximity bonus —
        # allow up to 2× base ceiling for generality
        assert obs.reward <= ceiling * 2 + 1e-9

    def test_step_reward_zero_when_no_satellites(self, env):
        for s in env.satellites:
            s["status"] = "destroyed"
        obs = env.step(KesslerAction(burns=[]))
        assert obs.reward == 0.0

    def test_partial_survival_reward_proportional(self, env):
        """1 out of 3 satellites alive should give 1/3 of full step reward.

        Uses deepcopy so both measurements run on the same task.
        env.reset() would increment episode_count, switching to Task 2
        (ECO-STATION) whose fuel multiplier breaks the 1/3 ratio.
        """
        import copy

        env_one = copy.deepcopy(env)
        env_one.satellites[1]["status"] = "destroyed"
        env_one.satellites[2]["status"] = "destroyed"
        obs_one = env_one.step(KesslerAction(burns=[]))

        env_all = copy.deepcopy(env)
        obs_all = env_all.step(KesslerAction(burns=[]))

        ratio = obs_one.reward / obs_all.reward
        assert abs(ratio - (1 / NUM_SATELLITES)) < 1e-9