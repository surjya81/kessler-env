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
GM = 1000.0          # Gravitational constant * Earth Mass (simplified)
DT = 1.0             # Time step
COLLISION_DIST = 2.0 # Proximity threshold for destruction
EARTH_RADIUS = 20.0  # Atmospheric crash limit
MAX_STEPS = 200


class KesslerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.satellites = []
        self.debris =[]
        self.cumulative_reward = 0.0

    def _generate_circular_orbit(self, radius: float):
        angle = np.random.uniform(0, 2 * math.pi)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        v = math.sqrt(GM / radius) # Orbital velocity formula
        vx = -v * math.sin(angle)
        vy = v * math.cos(angle)
        return x, y, vx, vy

    def reset(self) -> KesslerObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.cumulative_reward = 0.0
        self.satellites =[]
        
        # Spawn 3 controlled satellites
        for i in range(3):
            r = np.random.uniform(50, 80)
            x, y, vx, vy = self._generate_circular_orbit(r)
            self.satellites.append({
                "id": i, "x": x, "y": y, "vx": vx, "vy": vy, 
                "fuel": 100.0, "status": "active"
            })

        # Spawn 30 pieces of space debris with slightly erratic orbits
        self.debris =[]
        for i in range(30):
            r = np.random.uniform(40, 90)
            x, y, vx, vy = self._generate_circular_orbit(r)
            vx += np.random.uniform(-0.5, 0.5)
            vy += np.random.uniform(-0.5, 0.5)
            self.debris.append({"id": i, "x": x, "y": y, "vx": vx, "vy": vy})

        obs = self._get_observation([])
        obs.done = False
        obs.reward = 0.0
        return obs

    def _apply_gravity(self, obj: dict) -> bool:
        """Calculates gravity vector. Returns False if object crashed into Earth."""
        r_sq = obj['x']**2 + obj['y']**2
        r = math.sqrt(r_sq)
        if r < EARTH_RADIUS:
            return False 
        
        a = -GM / r_sq
        ax = a * (obj['x'] / r)
        ay = a * (obj['y'] / r)
        
        obj['vx'] += ax * DT
        obj['vy'] += ay * DT
        obj['x'] += obj['vx'] * DT
        obj['y'] += obj['vy'] * DT
        return True

    def step(self, action: KesslerAction) -> KesslerObservation:  # type: ignore[override]
        self._state.step_count += 1
        alerts =[]
        step_reward = 0.0

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
                    step_reward -= fuel_cost * 0.1 # Minor penalty for wasting fuel
                else:
                    alerts.append(f"Sat {sat['id']} failed burn: Insufficient fuel.")

        # 2. Physics & Gravity Update
        for sat in self.satellites:
            if sat['status'] == 'active':
                survived = self._apply_gravity(sat)
                if not survived:
                    sat['status'] = 'destroyed'
                    alerts.append(f"Sat {sat['id']} orbit decayed! Crashed into Earth.")
                    step_reward -= 500.0
                else:
                    step_reward += 1.0 # Base survival reward

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
                    step_reward -= 1000.0
                    
                    # The Kessler Effect: A collision creates 3 new pieces of debris
                    for _ in range(3):
                        self.debris.append({
                            "id": len(self.debris),
                            "x": sat['x'], "y": sat['y'],
                            "vx": sat['vx'] + np.random.uniform(-1, 1),
                            "vy": sat['vy'] + np.random.uniform(-1, 1)
                        })
                    break 

        self.cumulative_reward += step_reward
        
        # 4. Check Terminal Condition
        all_destroyed = all(s['status'] == 'destroyed' for s in self.satellites)
        is_done = all_destroyed or (self._state.step_count >= MAX_STEPS)

        obs = self._get_observation(alerts)
        obs.done = is_done
        obs.reward = step_reward
        return obs

    def _get_observation(self, alerts: list) -> KesslerObservation:
        return KesslerObservation(
            satellites=[SatelliteTelemetry(**s) for s in self.satellites],
            radar_debris=[DebrisTelemetry(**d) for d in self.debris],
            critical_alerts=alerts,
            total_score=self.cumulative_reward,
            done=False,
            reward=0.0
        )

    @property
    def state(self) -> State:
        return self._state