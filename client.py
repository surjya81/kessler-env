# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kessler Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import KesslerAction, KesslerObservation


class KesslerEnv(
    EnvClient[KesslerAction, KesslerObservation, State]
):
    """
    Client for the Kessler Env Environment.
    """

    def _step_payload(self, action: KesslerAction) -> Dict:
        """
        Convert KesslerAction to JSON payload for step message.
        """
        return {
            "burns":[
                {
                    "satellite_id": b.satellite_id,
                    "delta_vx": b.delta_vx,
                    "delta_vy": b.delta_vy
                }
                for b in action.burns
            ]
        }

    def _parse_result(self, payload: Dict) -> StepResult[KesslerObservation]:
        """
        Parse server response into StepResult[KesslerObservation].
        """
        obs_data = payload.get("observation", {})
        # Safely extract and fallback in case the server returns `null`
        reward_val = payload.get("reward")
        if reward_val is None:
            reward_val = obs_data.get("reward", 0.0)

        done_val = payload.get("done")
        if done_val is None:
            done_val = obs_data.get("done", False)

        observation = KesslerObservation(
            mission_objective=obs_data.get("mission_objective", ""),
            target_radius=obs_data.get("target_radius", 0.0),
            satellites=obs_data.get("satellites",[]),
            radar_debris=obs_data.get("radar_debris",[]),
            critical_alerts=obs_data.get("critical_alerts",[]),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            total_score=obs_data.get("total_score", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )