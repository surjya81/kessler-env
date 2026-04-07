# Copyright (c) Meta Platforms, Inc. and affiliates.
import math
import numpy as np
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import KesslerAction, KesslerObservation, ThrusterBurn, SatelliteTelemetry, DebrisTelemetry
except ImportError:
    from models import KesslerAction, KesslerObservation, ThrusterBurn, SatelliteTelemetry, DebrisTelemetry

# --- Orbital Constants ---
GM = 1000.0          
DT = 1.0             
COLLISION_DIST = 2.0 
EARTH_RADIUS = 20.0  
MAX_STEPS = 50       # Capped for 20-min hackathon inference limit
NUM_SATELLITES = 3

class KesslerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.satellites = []
        self.debris =[]

    def _generate_circular_orbit(self, radius: float):
        angle = np.random.uniform(0, 2 * math.pi)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        v = math.sqrt(GM / radius)
        return x, y, -v * math.sin(angle), v * math.cos(angle)

    def reset(self) -> KesslerObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.satellites =[]
        
        for i in range(NUM_SATELLITES):
            r = np.random.uniform(50, 80)
            x, y, vx, vy = self._generate_circular_orbit(r)
            self.satellites.append({
                "id": i, "x": x, "y": y, "vx": vx, "vy": vy, 
                "fuel": 100.0, "status": "active"
            })

        self.debris =[]
        for i in range(30):
            r = np.random.uniform(40, 90)
            x, y, vx, vy = self._generate_circular_orbit(r)
            self.debris.append({
                "id": i, "x": x, "y": y, 
                "vx": vx + np.random.uniform(-0.5, 0.5), 
                "vy": vy + np.random.uniform(-0.5, 0.5)
            })

        obs = self._get_observation([])
        obs.done = False
        obs.reward = 0.0
        return obs

    def _apply_gravity(self, obj: dict) -> bool:
        r_sq = obj['x']**2 + obj['y']**2
        r = math.sqrt(r_sq)
        if r < EARTH_RADIUS:
            return False 
        
        a = -GM / r_sq
        obj['vx'] += (a * (obj['x'] / r)) * DT
        obj['vy'] += (a * (obj['y'] / r)) * DT
        obj['x'] += obj['vx'] * DT
        obj['y'] += obj['vy'] * DT
        return True

    def step(self, action: KesslerAction) -> KesslerObservation:  # type: ignore[override]
        self._state.step_count += 1
        alerts =[]

        # 1. Apply Actions (Thruster burns)
        for burn in action.burns:
            sat = next((s for s in self.satellites if s['id'] == burn.satellite_id), None)
            if sat and sat['status'] == 'active':
                dvx = max(-1.0, min(1.0, burn.delta_vx))
                dvy = max(-1.0, min(1.0, burn.delta_vy))
                fuel_cost = math.sqrt(dvx**2 + dvy**2) * 5.0 
                
                if sat['fuel'] >= fuel_cost:
                    sat['vx'] += dvx
                    sat['vy'] += dvy
                    sat['fuel'] -= fuel_cost
                else:
                    alerts.append(f"Sat {sat['id']} failed burn: Insufficient fuel.")

        # 2. Physics & Gravity Update
        for sat in self.satellites:
            if sat['status'] == 'active':
                if not self._apply_gravity(sat):
                    sat['status'] = 'destroyed'
                    alerts.append(f"Sat {sat['id']} orbit decayed! Crashed into Earth.")

        for d in self.debris:
            self._apply_gravity(d)

        # 3. Collision Detection & Kessler Cascade
        for sat in self.satellites:
            if sat['status'] != 'active':
                continue
            for d in self.debris:
                dist = math.sqrt((sat['x'] - d['x'])**2 + (sat['y'] - d['y'])**2)
                if dist < COLLISION_DIST:
                    sat['status'] = 'destroyed'
                    alerts.append(f"CRITICAL: Sat {sat['id']} collided with Debris {d['id']}!")
                    # Cascade Effect
                    for _ in range(2):
                        self.debris.append({
                            "id": len(self.debris), "x": sat['x'], "y": sat['y'],
                            "vx": sat['vx'] + np.random.uniform(-1, 1),
                            "vy": sat['vy'] + np.random.uniform(-1, 1)
                        })
                    break 

        # 4. Calculate Step Reward (0.0 to 1.0 Normalization)
        # Agent gets a fractional reward for each satellite surviving this step.
        active_sats = sum(1 for s in self.satellites if s['status'] == 'active')
        step_reward = (active_sats / float(NUM_SATELLITES)) * (1.0 / float(MAX_STEPS))

        is_done = (active_sats == 0) or (self._state.step_count >= MAX_STEPS)

        obs = self._get_observation(alerts)
        obs.done = is_done
        obs.reward = step_reward
        return obs

    def _get_observation(self, alerts: list) -> KesslerObservation:
        return KesslerObservation(
            satellites=[SatelliteTelemetry(**s) for s in self.satellites],
            radar_debris=[DebrisTelemetry(**d) for d in self.debris],
            critical_alerts=alerts,
            done=False,
            reward=0.0
        )

    @property
    def state(self) -> State:
        return self._state