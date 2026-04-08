---
title: Kessler Env Environment Server
emoji: 🛰️
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - space-debris
  - orbital-mechanics
---

# 🛰️ Kessler Env — Orbital Debris Avoidance Environment

An [OpenEnv](https://github.com/facebookresearch/openenv)-compatible reinforcement learning environment where an LLM-powered agent acts as an **autonomous orbital traffic controller**, maneuvering satellites to survive Kessler Syndrome — a cascade of space debris collisions.

---

## 🌍 Environment Overview

The agent controls a fleet of **3 satellites** in low Earth orbit. At each step it receives radar telemetry and must issue thruster burns (or hold position) to avoid debris, conserve fuel, and complete mission-specific objectives.

The environment runs **3 sequential tasks** of escalating difficulty:

| Task | Name | Objective | Debris | Fuel |
|------|------|-----------|--------|------|
| 1 | **SURVIVAL** | Keep all satellites alive for 50 steps | 25 pieces | 100 units |
| 2 | **ECO-STATION** | Survive with minimal fuel usage — over-correction penalises score | 35 pieces | 50 units |
| 3 | **RENDEZVOUS** | Navigate Satellite 0 to orbital radius 100.0, survive rogue debris | 50 pieces | 100 units |

### Physics

- Newtonian gravity with `GM = 1000.0`, timestep `DT = 1.0`
- Satellites crash if they decay below `EARTH_RADIUS = 20.0`
- Collision distance: `2.0` units — triggers a **Kessler cascade** (spawns 2 new debris fragments)
- Hard mode includes random rogue debris spawns mid-episode

### Reward Structure

- **Base:** `(active_satellites / 3) × (1 / 50)` per step
- **ECO-STATION modifier:** multiplied by `current_fuel / initial_fuel` — burn less, score more
- **RENDEZVOUS bonus:** up to `+0.02` per step based on Sat 0's proximity to target radius 100.0
- **Episode score:** `sum(rewards) / 1.0`, clamped to `[0.0, 1.0]`
- **Success threshold:** score ≥ 0.1

---

## 🤖 Action & Observation Schema

### Action — `KesslerAction`

```json
{
  "burns": [
    { "satellite_id": 0, "delta_vx": 0.3, "delta_vy": -0.1 },
    { "satellite_id": 2, "delta_vx": 0.0, "delta_vy":  0.5 }
  ]
}
```

- `satellite_id`: integer, which satellite to maneuver
- `delta_vx`, `delta_vy`: velocity change, clamped to `[-1.0, 1.0]`
- Fuel cost per burn: `sqrt(dvx² + dvy²) × 5.0`
- Return `{"burns": []}` to hold position (free)

### Observation — `KesslerObservation`

```json
{
  "mission_objective": "SURVIVAL: Keep all satellites alive for 50 steps.",
  "target_radius": 0.0,
  "satellites": [
    { "id": 0, "x": 55.2, "y": -31.0, "vx": 3.1, "vy": 2.8, "fuel": 87.3, "status": "active" }
  ],
  "radar_debris": [
    { "id": 0, "x": 60.1, "y": -29.5, "vx": -2.9, "vy": 3.0 }
  ],
  "critical_alerts": [],
  "done": false,
  "reward": 0.02,
  "total_score": 0.14
}
```

---

## 🚀 Quick Start

### Prerequisites

Set the following environment variables (or create a `.env` file — see below):

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint (e.g. `https://api.openai.com/v1`) |
| `MODEL_NAME` | Model identifier (e.g. `gpt-4o`) |
| `HF_TOKEN` | Your Hugging Face / API key |
| `ENV_URL` | Running environment URL (default: `http://localhost:8000`) |

### Running Locally

**1. Start the environment server:**
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

**2. Configure credentials without hardcoding:**

Create a `.env` file in the project root (add `.env` to `.gitignore`):
```bash
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
HF_TOKEN=hf_your_token_here
ENV_URL=http://localhost:8000
```

Then run the inference script — it reads `.env` automatically via `python-dotenv`:
```bash
python inference.py
```

**3. Expected log output:**
```
[START] task=task_1_easy env=kessler-env model=gpt-4o
[STEP] step=1 action={'burns':[]} reward=0.02 done=false error=null
...
[END] success=true steps=50 score=0.820 rewards=0.02,0.02,...
```

### Connecting to a Running Space

```python
from kessler_env import KesslerEnv, KesslerAction

with KesslerEnv(base_url="https://<your-space>.hf.space") as env:
    obs = env.reset()
    for _ in range(50):
        result = env.step(KesslerAction(burns=[]))
        if result.observation.done:
            break
```

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/web` | GET | Interactive web UI |
| `/docs` | GET | OpenAPI / Swagger documentation |
| `/health` | GET | Container health check |
| `/reset` | POST | Reset environment, get initial observation |
| `/step` | POST | Execute an action, get next observation |
| `/state` | GET | Current environment state |
| `/schema` | GET | Action and observation JSON schemas |
| `/ws` | WebSocket | Persistent low-latency session |

### WebSocket Protocol

```python
import websockets, json, asyncio

async def run():
    async with websockets.connect("wss://<space-url>/ws") as ws:
        # Reset
        await ws.send(json.dumps({"type": "reset"}))
        obs = json.loads(await ws.recv())

        # Step
        action = {"burns": [{"satellite_id": 0, "delta_vx": 0.1, "delta_vy": 0.0}]}
        await ws.send(json.dumps({"type": "step", "data": action}))
        result = json.loads(await ws.recv())

        # Close
        await ws.send(json.dumps({"type": "close"}))

asyncio.run(run())
```

---

## 🏗️ Deployment

### Deploy to Hugging Face Spaces

```bash
# From the project root (where openenv.yaml lives)
openenv push

# Or with options
openenv push --repo-id my-org/kessler-env --private
```

Set `HF_TOKEN`, `API_BASE_URL`, and `MODEL_NAME` as **Space Secrets** in your repo's Settings → Variables and Secrets tab. They are injected as environment variables at runtime — no code changes needed.

### Docker

```bash
# Build
docker build -t kessler-env:latest -f server/Dockerfile .

# Run
docker run -p 8000:8000 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o \
  -e HF_TOKEN=hf_your_token \
  kessler-env:latest
```

---

## 📁 Project Structure

```
kessler_env/
├── openenv.yaml                    # OpenEnv spec manifest
├── inference.py                    # LLM agent inference script (entry point)
├── models.py                       # KesslerAction & KesslerObservation schemas
├── Dockerfile                      # Container definition
├── README.md                       # This file
├── .dockerignore                   # Docker build exclusions
├── __init__.py                     # Module exports
├── pyproject.toml                  # Project metadata and dependencies
├── uv.lock                         # Locked dependencies (generated)
├── client.py                       # KesslerEnv client
└── server/
    ├── __init__.py                 # Server module exports
    ├── app.py                      # FastAPI app (HTTP + WebSocket)
    ├── kessler_env_environment.py  # Core physics & reward logic
    └── requirements                 
```

---

## ⚙️ openenv.yaml

```yaml
spec_version: 1
name: kessler_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

---

## 📋 Inference Script Requirements

The `inference.py` must be in the **project root** and emit structured stdout logs in this exact format:

```
[START] task=<name> env=<benchmark> model=<model>
[STEP] step=<n> action=<json> reward=<float> done=<bool> error=<str|null>
[END] success=<bool> steps=<n> score=<float> rewards=<comma-list>
```

The script runs **3 tasks sequentially** (`task_1_easy`, `task_2_medium`, `task_3_hard`)

---
