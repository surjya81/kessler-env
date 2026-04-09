# Copyright (c) Meta Platforms, Inc. and affiliates.
import math
import numpy as np
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import KesslerAction, KesslerObservation, ThrusterBurn, SatelliteTelemetry, DebrisTelemetry
except ImportError:
    from models import KesslerAction, KesslerObservation, ThrusterBurn, SatelliteTelemetry, DebrisTelemetry

try:
    from logger import get_logger
except ImportError:
    from .logger import get_logger  # type: ignore

logger = get_logger(__name__)

# --- Orbital Constants ---
GM = 1000.0
DT = 1.0
COLLISION_DIST = 2.0
EARTH_RADIUS = 20.0
MAX_STEPS = 50       # Capped for 20-min hackathon inference limit
NUM_SATELLITES = 3

# Epsilon used to keep per-episode score strictly inside (0, 1).
# The environment adds this as a floor so that even a fully-destroyed run
# emits a non-zero score, and the ceiling prevents a perfect run from
# reaching exactly 1.0.
_SCORE_EPSILON = 1e-3


class KesslerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.satellites = []
        self.debris = []
        self.episode_count = 0  # Tracks difficulty tier
        logger.info("KesslerEnvironment initialised")

    def _generate_circular_orbit(self, radius: float):
        angle = np.random.uniform(0, 2 * math.pi)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        v = math.sqrt(GM / radius)
        return x, y, -v * math.sin(angle), v * math.cos(angle)

    def reset(self) -> KesslerObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.satellites = []

        # Escalate difficulty on each reset (0: Easy, 1: Medium, 2: Hard)
        self.current_task_idx = self.episode_count % 3
        self.episode_count += 1

        if self.current_task_idx == 0:
            self.mission_objective = "SURVIVAL: Keep all satellites alive for 50 steps."
            num_debris, fuel = 25, 100.0
            self.target_radius = 0.0
            vel_variance, self.rogue_chance = 0.0, 0.0
        elif self.current_task_idx == 1:
            self.mission_objective = "ECO-STATION: Survive, but fuel usage heavily reduces your score. Do not over-correct."
            num_debris, fuel = 35, 50.0
            self.target_radius = 0.0
            vel_variance, self.rogue_chance = 0.8, 0.0
        else:
            self.mission_objective = "RENDEZVOUS: Navigate Satellite 0 to reach and hold an orbital radius of exactly 100.0."
            num_debris, fuel = 50, 100.0
            self.target_radius = 100.0
            vel_variance, self.rogue_chance = 1.0, 0.06

        logger.info(
            "reset() — episode=%d task_idx=%d objective=%r debris=%d fuel=%.1f",
            self.episode_count, self.current_task_idx, self.mission_objective,
            num_debris, fuel,
        )

        for i in range(NUM_SATELLITES):
            r = np.random.uniform(50, 80)
            x, y, vx, vy = self._generate_circular_orbit(r)
            self.satellites.append({
                "id": i, "x": x, "y": y, "vx": vx, "vy": vy,
                "fuel": fuel, "status": "active"
            })
            logger.debug("  Sat %d spawned at (%.2f, %.2f) v=(%.2f, %.2f) r=%.1f", i, x, y, vx, vy, r)

        self.debris = []
        for i in range(num_debris):
            r = np.random.uniform(40, 90)
            x, y, vx, vy = self._generate_circular_orbit(r)
            self.debris.append({
                "id": i, "x": x, "y": y,
                "vx": vx + np.random.uniform(-vel_variance, vel_variance),
                "vy": vy + np.random.uniform(-vel_variance, vel_variance)
            })

        logger.debug("reset() complete — %d debris spawned", len(self.debris))

        obs = self._get_observation([])
        obs.done = False
        obs.reward = 0.0
        return obs

    def _apply_gravity(self, obj: dict) -> bool:
        r_sq = obj['x'] ** 2 + obj['y'] ** 2
        r = math.sqrt(r_sq)
        if r < EARTH_RADIUS:
            logger.debug("Object at (%.2f, %.2f) inside Earth radius — marking destroyed", obj['x'], obj['y'])
            return False

        a = -GM / r_sq
        obj['vx'] += (a * (obj['x'] / r)) * DT
        obj['vy'] += (a * (obj['y'] / r)) * DT
        obj['x'] += obj['vx'] * DT
        obj['y'] += obj['vy'] * DT
        return True

    def step(self, action: KesslerAction) -> KesslerObservation:  # type: ignore[override]
        self._state.step_count += 1
        step = self._state.step_count
        alerts = []

        logger.debug("step %d — burns received: %d", step, len(action.burns))

        # 1. Apply Actions (Thruster burns)
        for burn in action.burns:
            sat = next((s for s in self.satellites if s['id'] == burn.satellite_id), None)
            if sat and sat['status'] == 'active':
                dvx = max(-1.0, min(1.0, burn.delta_vx))
                dvy = max(-1.0, min(1.0, burn.delta_vy))
                fuel_cost = math.sqrt(dvx ** 2 + dvy ** 2) * 5.0

                if sat['fuel'] >= fuel_cost:
                    sat['vx'] += dvx
                    sat['vy'] += dvy
                    sat['fuel'] -= fuel_cost
                    logger.debug(
                        "  Sat %d burn dvx=%.3f dvy=%.3f fuel_cost=%.2f remaining=%.2f",
                        sat['id'], dvx, dvy, fuel_cost, sat['fuel'],
                    )
                else:
                    alerts.append(f"Sat {sat['id']} failed burn: Insufficient fuel.")
                    logger.warning("step %d — Sat %d burn failed: insufficient fuel (%.2f < %.2f)",
                                   step, sat['id'], sat['fuel'], fuel_cost)
            else:
                logger.debug("  burn targeting unknown/inactive sat id=%d — skipped", burn.satellite_id)

        # 2. Hostile Mechanic: Random Rogue Debris (Hard Mode Only)
        if random.random() < self.rogue_chance:
            r = 95.0
            x, y, vx, vy = self._generate_circular_orbit(r)
            self.debris.append({
                "id": len(self.debris),
                "x": x, "y": y,
                "vx": vx * 1.2,
                "vy": vy * 1.2
            })
            alerts.append("WARNING: High-velocity rogue debris entered radar!")
            logger.info("step %d — rogue debris spawned (total debris=%d)", step, len(self.debris))

        # 3. Physics & Gravity Update
        for sat in self.satellites:
            if sat['status'] == 'active':
                if not self._apply_gravity(sat):
                    sat['status'] = 'destroyed'
                    alerts.append(f"Sat {sat['id']} orbit decayed! Crashed into Earth.")
                    logger.warning("step %d — Sat %d orbit decayed (crashed into Earth)", step, sat['id'])

        for d in self.debris:
            self._apply_gravity(d)

        # 4. Collision Detection & Kessler Cascade
        for sat in self.satellites:
            if sat['status'] != 'active':
                continue
            for d in self.debris:
                dist = math.sqrt((sat['x'] - d['x']) ** 2 + (sat['y'] - d['y']) ** 2)
                if dist < COLLISION_DIST:
                    sat['status'] = 'destroyed'
                    alerts.append(f"CRITICAL: Sat {sat['id']} collided with Debris {d['id']}!")
                    logger.warning(
                        "step %d — COLLISION Sat %d <-> Debris %d dist=%.3f",
                        step, sat['id'], d['id'], dist,
                    )
                    # Cascade Effect — spawn 2 new debris fragments
                    for _ in range(2):
                        self.debris.append({
                            "id": len(self.debris), "x": sat['x'], "y": sat['y'],
                            "vx": sat['vx'] + np.random.uniform(-1, 1),
                            "vy": sat['vy'] + np.random.uniform(-1, 1)
                        })
                    break

        # 5. Calculate Step Reward
        #
        # Rewards are scaled so the maximum possible cumulative score across all
        # MAX_STEPS steps is (1 - 2*_SCORE_EPSILON), keeping the final episode
        # score strictly inside the open interval (0, 1) as required by the grader.
        #
        # The floor is handled in inference.py by adding _SCORE_EPSILON after
        # summing rewards, ensuring even a 0-reward run produces a score > 0.
        active_sats = sum(1 for s in self.satellites if s['status'] == 'active')

        # Base survival reward
        step_reward = (active_sats / float(NUM_SATELLITES)) * ((1.0 - 2 * _SCORE_EPSILON) / float(MAX_STEPS))

        # Task 1 – ECO-STATION: heavy fuel penalty
        if self.current_task_idx == 1:
            current_fuel = sum(s.get('fuel', 0.0) for s in self.satellites if s['status'] == 'active')
            initial_fuel = NUM_SATELLITES * 50.0
            fuel_frac = max(0.0, current_fuel / initial_fuel)
            step_reward *= fuel_frac
            logger.debug(
                "step %d ECO-STATION fuel_frac=%.4f step_reward=%.6f",
                step, fuel_frac, step_reward,
            )

        # Task 2 – RENDEZVOUS: proximity bonus for Sat 0
        elif self.current_task_idx == 2:
            step_reward /= 2.0  # Shrink base to accommodate for proximity bonus
            sat0 = next((s for s in self.satellites if s['id'] == 0 and s['status'] == 'active'), None)
            if sat0:
                r0 = math.sqrt(sat0['x'] ** 2 + sat0['y'] ** 2)
                proximity = max(0.0, 1.0 - abs(r0 - self.target_radius) / 30.0)
                # Extra reward capped at the same per-step budget
                step_reward += (proximity * ((1.0 - 2 * _SCORE_EPSILON) / float(MAX_STEPS))) / 2.0
                logger.debug(
                    "step %d RENDEZVOUS sat0_r=%.2f target=%.2f proximity=%.4f step_reward=%.6f",
                    step, r0, self.target_radius, proximity, step_reward,
                )

        is_done = (active_sats == 0) or (self._state.step_count >= MAX_STEPS)

        logger.info(
            "step %d — active_sats=%d/%d step_reward=%.6f done=%s alerts=%d",
            step, active_sats, NUM_SATELLITES, step_reward, is_done, len(alerts),
        )

        obs = self._get_observation(alerts)
        obs.done = is_done
        obs.reward = step_reward
        return obs

    def _get_observation(self, alerts: list) -> KesslerObservation:
        return KesslerObservation(
            mission_objective=self.mission_objective,
            target_radius=self.target_radius,
            satellites=[SatelliteTelemetry(**s) for s in self.satellites],
            radar_debris=[DebrisTelemetry(**d) for d in self.debris],
            critical_alerts=alerts,
            done=False,
            reward=0.0
        )

    @property
    def state(self) -> State:
        return self._state


# Singleton setup
_SINGLETON: KesslerEnvironment | None = None


def get_instance() -> KesslerEnvironment:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = KesslerEnvironment()
    return _SINGLETON