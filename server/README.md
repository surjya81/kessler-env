# 🛰️ Kessler Env — Web Interface & API Guide

This guide is for **humans** who want to interact with the Kessler Env environment directly — exploring the simulation, triggering episodes manually, and watching what happens — without writing an inference script.

---

## Getting Here

Once the server is running (locally or on HF Spaces), open your browser and go to:

```
http://localhost:8000/web
```

You'll land on the interactive web UI. From here you can reset the environment, send actions, and read observations — all through a point-and-click interface.

If you're on HF Spaces, replace `localhost:8000` with your Space URL (e.g. `https://my-org-kessler-env.hf.space/web`).

---

## What You're Looking At

The environment simulates **3 satellites** in low Earth orbit, surrounded by a debris field. Each step, you decide whether to fire any satellite's thrusters. The physics run, collisions are checked, and you get back updated telemetry.

There are three task types in rotation, each harder than the last:

| Task | Name | What it asks of you |
|------|------|---------------------|
| **SURVIVAL** | Keep all 3 satellites alive for 50 steps | Dodge the debris field — don't run out of positions to hide |
| **ECO-STATION** | Survive *and* conserve fuel | Every unnecessary burn tanks your score — only fire when you have to |
| **RENDEZVOUS** | Navigate Satellite 0 to orbital radius 100.0 | Hold the target orbit against 50 debris pieces and rogue spawns |

---

## Interactive Web UI (`/web`)

The web interface at `/web` lets you:

- **Reset** the environment to start a fresh episode
- **View the current observation** — satellite positions, velocities, fuel levels, radar debris, and any active alerts
- **Submit a thruster action** using a form — choose which satellite to burn and by how much
- **Step through the episode** one turn at a time and watch the reward accumulate

It's the fastest way to get a feel for what the agent is actually dealing with.

---

## REST API

If you want to drive the environment from a terminal, a notebook, or any HTTP client (curl, Postman, Python `requests`), the following endpoints are available.

### `POST /reset`

Starts a new episode. The environment picks the next task in the rotation (Easy → Medium → Hard → Easy…) and returns the initial observation.

**Request:** No body needed.

```bash
curl -X POST http://localhost:8000/reset
```

**Response:**

```json
{
  "mission_objective": "SURVIVAL: Keep all satellites alive for 50 steps.",
  "target_radius": 0.0,
  "satellites": [
    { "id": 0, "x": 55.2, "y": -31.0, "vx": 3.1, "vy": 2.8, "fuel": 100.0, "status": "active" },
    { "id": 1, "x": -42.7, "y": 61.5, "vx": -3.8, "vy": -2.6, "fuel": 100.0, "status": "active" },
    { "id": 2, "x": 10.3, "y": -70.2, "vx": 4.2, "vy": 0.9, "fuel": 100.0, "status": "active" }
  ],
  "radar_debris": [
    { "id": 0, "x": 60.1, "y": -29.5, "vx": -2.9, "vy": 3.0 },
    { "id": 3, "x": 58.8, "y": -35.1, "vx": -3.1, "vy": 2.7 }
  ],
  "radar_range": 0.0,
  "critical_alerts": [],
  "done": false,
  "reward": 0.0,
  "total_score": 0.0
}
```

**Reading the response:**

- `satellites` — your controllable fleet. `status` is `"active"` or `"destroyed"`. `fuel` starts at 100 (or 50 on ECO-STATION).
- `radar_debris` — every tracked debris object within sensor range. Each has a position (`x`, `y`) and velocity (`vx`, `vy`) — no fuel, because debris doesn't maneuver.
- `radar_range` — your sensor horizon. `0.0` means you can see everything. A positive value means debris further than that distance from any of your satellites is invisible to you.
- `critical_alerts` — warnings emitted this step (collisions, fuel failures, rogue debris spawns, judge feedback).
- `mission_objective` — the plain-English goal for this episode.
- `target_radius` — only non-zero on RENDEZVOUS. This is the orbital radius Satellite 0 must reach and hold.

---

### `POST /step`

Advances the simulation by one timestep. You provide thruster burns for any satellites you want to maneuver; satellites you omit hold their current trajectory.

**Request body:**

```json
{
  "burns": [
    { "satellite_id": 0, "delta_vx": 0.3, "delta_vy": -0.1 }
  ]
}
```

To hold position (do nothing), send an empty burns list:

```json
{ "burns": [] }
```

**Field constraints:**
- `satellite_id` — must match one of the `id` values from the current observation (0, 1, or 2).
- `delta_vx`, `delta_vy` — velocity change in position units per timestep. **Clamped to [-1.0, 1.0].**
- Fuel cost per burn: `sqrt(delta_vx² + delta_vy²) × 5.0`. A full-thrust burn on both axes costs ~7.07 fuel. Holding (`burns: []`) is free.
- Burns on a destroyed satellite or a satellite with insufficient fuel are silently ignored (you'll see an alert).

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"burns": [{"satellite_id": 0, "delta_vx": 0.3, "delta_vy": -0.1}]}'
```

**Response** — same shape as `/reset`, plus updated values:

```json
{
  "mission_objective": "SURVIVAL: Keep all satellites alive for 50 steps.",
  "target_radius": 0.0,
  "satellites": [
    { "id": 0, "x": 58.4, "y": -28.3, "vx": 3.4, "vy": 2.7, "fuel": 93.4, "status": "active" },
    ...
  ],
  "radar_debris": [ ... ],
  "radar_range": 0.0,
  "critical_alerts": [],
  "done": false,
  "reward": 0.0193,
  "total_score": 0.0193
}
```

**Reading the step response:**

- `reward` — the reward earned *this step* (higher is better). On ECO-STATION it's scaled by remaining fuel fraction. On RENDEZVOUS it includes a proximity bonus for Satellite 0.
- `total_score` — cumulative reward across all steps so far.
- `done` — `true` when all satellites are destroyed OR 50 steps have elapsed. Once `done` is `true`, call `/reset` to start fresh.
- `critical_alerts` — check this every step. Collision warnings give you one last chance to understand what went wrong. If the LLM judge is enabled, its reasoning also appears here.

---

### `GET /state`

Returns the current episode metadata — useful for checking where you are in an episode without triggering any physics.

```bash
curl http://localhost:8000/state
```

**Response:**

```json
{
  "episode_id": "a3f2c1d8-9e44-4b2a-bf11-7c3d2e1a0f55",
  "step_count": 12
}
```

- `episode_id` — a UUID generated fresh on each `/reset`. Useful for logging.
- `step_count` — how many `/step` calls have been made this episode. The episode ends when this reaches 50.

---

### `GET /schema`

Returns the full JSON Schema for both the action and observation types. Useful if you're building a client and want to validate your payloads before sending them.

```bash
curl http://localhost:8000/schema
```

**Response:**

```json
{
  "action_schema": { ... },
  "observation_schema": { ... }
}
```

---

### `GET /health`

A lightweight liveness check. Returns `200 OK` if the server is up.

```bash
curl http://localhost:8000/health
```

---

### `GET /docs`

OpenAPI / Swagger documentation — auto-generated from the FastAPI app. Open this in a browser for an interactive API explorer where you can fill in request bodies and run them directly from the UI.

```
http://localhost:8000/docs
```

---

## WebSocket — Persistent Sessions

The REST endpoints create a new environment instance per call, which is fine for casual exploration. For running a full episode (50 steps) without overhead, use the WebSocket endpoint instead. It keeps a single stateful session open.

```
ws://localhost:8000/ws
```

The protocol is simple: send JSON with a `type` field, receive JSON back.

### Reset

```json
→ { "type": "reset" }
← { "type": "observation", "data": { ...KesslerObservation... } }
```

### Step

```json
→ { "type": "step", "data": { "burns": [ {"satellite_id": 0, "delta_vx": 0.1, "delta_vy": 0.0} ] } }
← { "type": "observation", "data": { ...KesslerObservation... } }
```

### Close

```json
→ { "type": "close" }
← { "type": "closed" }
```

### Error responses

If the server encounters a problem with your message (bad JSON, unknown type, invalid action), it responds with:

```json
{ "type": "error", "data": { "message": "description of what went wrong" } }
```

### Full Python example

```python
import asyncio
import json
import websockets

async def play_episode():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        # Start a new episode
        await ws.send(json.dumps({"type": "reset"}))
        response = json.loads(await ws.recv())
        obs = response["data"]
        print("Mission:", obs["mission_objective"])

        for step in range(50):
            # Decide your action — this example just holds position
            action = {"burns": []}

            await ws.send(json.dumps({"type": "step", "data": action}))
            response = json.loads(await ws.recv())
            obs = response["data"]

            print(f"Step {step + 1} | reward={obs['reward']:.4f} | done={obs['done']}")

            if obs.get("critical_alerts"):
                print("  Alerts:", obs["critical_alerts"])

            if obs["done"]:
                print(f"Episode over — total score: {obs['total_score']:.3f}")
                break

        await ws.send(json.dumps({"type": "close"}))

asyncio.run(play_episode())
```

---

## Tips for Manual Exploration

**Start with Task 1 (SURVIVAL).** Call `/reset` until `mission_objective` says `SURVIVAL`. The debris field is smaller and fuel is plentiful — a good place to learn what the observation looks like before debris starts hitting.

**Watch `critical_alerts` carefully.** A `CRITICAL: Sat X collided with Debris Y!` alert means that satellite is now `destroyed` and two new fragment pieces have spawned at its last position. You can't un-destroy a satellite, but surviving ones can still score.

**Hold before you burn.** On ECO-STATION especially, the safest first experiment is sending `{"burns": []}` every step and watching whether satellites survive on their own. If they do, you've just scored the maximum possible fuel bonus.

**On RENDEZVOUS, check `target_radius`.** It will be `100.0`. Look at Satellite 0's current `x` and `y` — its current orbital radius is `sqrt(x² + y²)`. The gap between that and 100.0 tells you roughly how hard you need to push.

**If `radar_range` is non-zero**, be aware that the `radar_debris` list is incomplete. Debris outside the cone is still there — it just won't appear until it drifts into sensor range. Plan ahead.

---

## Environment Behaviour Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max steps per episode | 50 | Episode auto-ends at step 50 |
| Number of satellites | 3 | IDs 0, 1, 2 |
| Collision distance | 2.0 units | Any debris within this range destroys the satellite |
| Earth radius (danger zone) | 20.0 units | Satellites that decay below this are destroyed |
| Thruster delta clamp | ±1.0 | Both `delta_vx` and `delta_vy` are clamped before applying |
| Fuel cost formula | `sqrt(dvx² + dvy²) × 5.0` | Diagonal burns cost more than axis-aligned ones |
| Cascade fragments | 2 per collision | Each collision spawns 2 new debris pieces at the impact site |
| Score range | (0.001, 0.999) | Strictly inside (0, 1); success threshold is 0.1 |