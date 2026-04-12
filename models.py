# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Kessler Env Environment.
"""
import json
from pydantic import BaseModel, Field, field_validator
from typing import List
from openenv.core.env_server.types import Action, Observation


class ThrusterBurn(BaseModel):
    satellite_id: int = Field(..., description="ID of the satellite to maneuver")
    delta_vx: float = Field(..., description="Change in X velocity (range: -1.0 to 1.0)")
    delta_vy: float = Field(..., description="Change in Y velocity (range: -1.0 to 1.0)")


class KesslerAction(Action):
    """Action for the Kessler environment - assigning thruster burns to dodge debris."""
    burns: List[ThrusterBurn] = Field(
        default_factory=list,
        description="List of thruster maneuvers to execute. Leave empty for no burns."
    )
    # VALIDATOR FOR WEB INTERFACE COMPATIBILITY:
    @field_validator('burns', mode='before')
    def parse_burns_from_string(cls, v):
        if isinstance(v, str):
            try:
                data = json.loads(v)
                # If the UI wrapped it in {"action": {"burns": [...]}}
                if isinstance(data, dict):
                    if "action" in data:
                        data = data["action"]
                    if "burns" in data:
                        return data["burns"]
                return data
            except json.JSONDecodeError:
                raise ValueError("Input must be a valid JSON string representing a list of burns.")
        return v

class SatelliteTelemetry(BaseModel):
    id: int
    x: float
    y: float
    vx: float
    vy: float
    fuel: float
    status: str = Field(..., description="'active' or 'destroyed'")


class DebrisTelemetry(BaseModel):
    id: int
    x: float
    y: float
    vx: float
    vy: float


class KesslerObservation(Observation):
    """Observation returning orbital telemetry and alerts."""
    mission_objective: str = Field(default="", description="The goal for the current episode")
    target_radius: float = Field(default=0.0, description="If greater than 0, navigate Sat 0 to this orbital radius")
    satellites: List[SatelliteTelemetry] = Field(..., description="Telemetry of your controlled satellites")
    radar_debris: List[DebrisTelemetry] = Field(
        ...,
        description=(
            "Debris within radar_range of at least one active satellite. "
            "Objects outside this range are not visible — plan accordingly."
        )
    )
    radar_range: float = Field(
        default=0.0,
        description=(
            "Sensor horizon in position units. Only debris within this distance "
            "of any active satellite appears in radar_debris. "
            "0.0 means unlimited visibility (full observability mode)."
        )
    )
    critical_alerts: List[str] = Field(default_factory=list, description="Alerts such as collisions or low fuel")
    done: bool = False
    reward: float = 0.0
    total_score: float = Field(default=0.0, description="Cumulative mission score")