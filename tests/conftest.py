"""
conftest.py — shared fixtures for the Kessler Env test suite.

The tests mock out openenv.core so they can run without the full
openenv package installed. All environment logic is tested directly
against KesslerEnvironment.
"""
import math
import sys
import types
import pytest # pyright: ignore[reportMissingImports]


# ---------------------------------------------------------------------------
# Stub out openenv.core so imports in models.py / environment.py don't fail
# ---------------------------------------------------------------------------
def _make_openenv_stubs():
    """Create minimal stub modules for openenv.core."""

    from pydantic import BaseModel

    # Action / Observation must inherit from BaseModel so that subclasses
    # like KesslerAction and KesslerObservation are proper Pydantic models.
    # A plain class here causes TypeError: takes no arguments when constructing
    # them with keyword arguments in tests.
    class Action(BaseModel):
        class Config:
            extra = "allow"

    class Observation(BaseModel):
        class Config:
            extra = "allow"

    class State:
        def __init__(self, episode_id=None, step_count=0):
            self.episode_id = episode_id
            self.step_count = step_count

    class Environment:
        pass

    class StepResult:
        def __init__(self, observation=None, reward=None, done=False):
            self.observation = observation
            self.reward = reward
            self.done = done

    # Stub module tree: openenv.core.env_server.types, .interfaces, etc.
    openenv          = types.ModuleType("openenv")
    openenv_core     = types.ModuleType("openenv.core")
    openenv_server   = types.ModuleType("openenv.core.env_server")
    openenv_types    = types.ModuleType("openenv.core.env_server.types")
    openenv_iface    = types.ModuleType("openenv.core.env_server.interfaces")
    openenv_client   = types.ModuleType("openenv.core.client_types")

    openenv_types.Action      = Action
    openenv_types.Observation = Observation
    openenv_types.State       = State
    openenv_iface.Environment = Environment
    openenv_client.StepResult = StepResult

    # EnvClient stub (used by client.py)
    class EnvClient:
        def __class_getitem__(cls, item):
            return cls

    openenv_core.EnvClient = EnvClient

    sys.modules["openenv"]                          = openenv
    sys.modules["openenv.core"]                     = openenv_core
    sys.modules["openenv.core.env_server"]          = openenv_server
    sys.modules["openenv.core.env_server.types"]    = openenv_types
    sys.modules["openenv.core.env_server.interfaces"] = openenv_iface
    sys.modules["openenv.core.client_types"]        = openenv_client


_make_openenv_stubs()

# Now we can safely import from the project.
# Insert both the project root (for models.py, logger.py) and server/
# (for kessler_env_environment.py) so imports resolve regardless of where
# pytest is invoked from.
import os as _os
_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_server = _os.path.join(_root, "server")
sys.path.insert(0, _server)
sys.path.insert(0, _root)

from models import KesslerAction, KesslerObservation, ThrusterBurn  # noqa: E402
import kessler_env_environment as env_module                         # noqa: E402 # pyright: ignore[reportMissingImports]
from kessler_env_environment import KesslerEnvironment               # noqa: E402 # pyright: ignore[reportMissingImports]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def env():
    """Fresh KesslerEnvironment instance, reset to Task 1 (SURVIVAL)."""
    # Patch episode_count so reset() always picks task_1 (index 0)
    e = KesslerEnvironment.__new__(KesslerEnvironment)
    from openenv.core.env_server.types import State
    from uuid import uuid4
    e._state = State(episode_id=str(uuid4()), step_count=0)
    e.satellites = []
    e.debris = []
    e.episode_count = 0
    e.reset()
    return e


@pytest.fixture()
def env_task2():
    """Environment reset to Task 2 (ECO-STATION)."""
    e = KesslerEnvironment.__new__(KesslerEnvironment)
    from openenv.core.env_server.types import State
    from uuid import uuid4
    e._state = State(episode_id=str(uuid4()), step_count=0)
    e.satellites = []
    e.debris = []
    e.episode_count = 1   # episode_count % 3 == 1 → task_2
    e.reset()
    return e


@pytest.fixture()
def env_task3():
    """Environment reset to Task 3 (RENDEZVOUS)."""
    e = KesslerEnvironment.__new__(KesslerEnvironment)
    from openenv.core.env_server.types import State
    from uuid import uuid4
    e._state = State(episode_id=str(uuid4()), step_count=0)
    e.satellites = []
    e.debris = []
    e.episode_count = 2   # episode_count % 3 == 2 → task_3
    e.reset()
    return e


@pytest.fixture()
def no_burn():
    """KesslerAction with no burns — hold position."""
    return KesslerAction(burns=[])


@pytest.fixture(autouse=True)
def reset_radar_range():
    """
    Ensure RADAR_RANGE is restored to 0.0 (full observability) after each
    test that patches it, so tests don't bleed into each other.
    """
    original = env_module.RADAR_RANGE
    yield
    env_module.RADAR_RANGE = original