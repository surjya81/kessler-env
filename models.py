# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Kessler Env Environment.

The kessler_env environment is a simple test environment that echoes back messages.
"""
from pydantic import BaseModel, Field
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
    satellites: List[SatelliteTelemetry] = Field(..., description="Telemetry of your controlled satellites")
    radar_debris: List[DebrisTelemetry] = Field(..., description="Tracked debris locations and velocities")
    critical_alerts: List[str] = Field(default_factory=list, description="Alerts such as collisions or low fuel")
    total_score: float = Field(default=0.0, description="Cumulative mission score")
