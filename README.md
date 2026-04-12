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

# 🛰️ Kessler Env — Teaching LLMs to Manage Orbital Traffic

### Can a language model learn to keep satellites alive in a debris field it's never seen before?

We put an LLM in the seat of an orbital traffic controller. No pre-training on conjunction analysis. No lookup tables. Just radar telemetry, a fuel budget, and the clock ticking. It has to figure out when to burn, when to hold, and how not to touch the joystick so much that it runs dry trying to dodge shadows.

**This is Kessler Env** — a physics-driven orbital mechanics simulation where an agent learns to manage real tradeoffs: survive debris cascades, conserve fuel, and navigate to precise orbital targets. Built with [OpenEnv](https://github.com/facebookresearch/openenv), deployed on HF Spaces.

---

## Why This Problem Is Worth Solving

Space situational awareness (SSA) is genuinely hard. Right now, operators at agencies like ESA and the US Space Force receive **Conjunction Data Messages (CDMs)** — automated warnings that two objects may collide — and have to decide: burn or don't burn? Maneuver now, or wait for better tracking data?

That decision has real consequences. A burn costs fuel (which is finite and irreplaceable on orbit). A missed collision can trigger a debris cascade — the exact **Kessler Syndrome** this environment is named after — where one collision creates enough fragments to cause more collisions, until an entire orbital band becomes unusable.

The core question this environment is designed to answer: **can we train a model to reason about that tradeoff from telemetry alone?**

It won't replace a flight dynamics team tomorrow. But as a training and evaluation benchmark for orbital decision-making agents, it's a more grounded starting point than most RL environments — because the physics, the constraints, and the failure modes are real.

---

## What the Environment Models

The agent controls a fleet of **3 satellites** in low Earth orbit. At each timestep it receives radar telemetry for all its satellites and every tracked debris piece, then decides whether to fire thrusters.

Three tasks of escalating difficulty test genuinely different skills:

| Task | Name | Core Challenge | Debris | Fuel |
|------|------|---------------|--------|------|
| 1 | **SURVIVAL** | Keep all satellites alive for 50 steps — just don't die | 25 pieces | 100 units |
| 2 | **ECO-STATION** | Survive, but over-correcting tanks your score — conservation matters | 35 pieces | 50 units |
| 3 | **RENDEZVOUS** | Navigate Satellite 0 to hold an orbital radius of exactly 100.0 against rogue debris | 50 pieces + spawns | 100 units |

These aren't arbitrarily tiered. Each task isolates a different real-world skill:
- Task 1 tests basic threat recognition and avoidance
- Task 2 forces the agent to reason about *whether* a burn is worth the fuel cost, not just *how* to dodge
- Task 3 adds the rendezvous planning problem on top — the agent has to hold a target orbit while debris comes at it

### Physics

The simulation runs Newtonian orbital mechanics per timestep:

- Gravity: `GM = 1000.0`, timestep `DT = 1.0`
- Satellites that decay below `EARTH_RADIUS = 20.0` are destroyed
- Collision threshold: `2.0` units — any contact with debris destroys the satellite and triggers a **Kessler cascade** (two new fragment pieces spawn from the wreckage)
- Hard mode (Task 3) includes random rogue debris that spawns mid-episode at the radar edge with boosted velocity

### Reward Shaping

Rewards are designed to produce clean signal for GRPO training — not just binary win/lose:

```
Base (all tasks):   (active_sats / 3) × ((1 - 2ε) / 50) per step
ECO modifier:       × (current_fuel / initial_fuel)  — burn less, score more
RENDEZVOUS bonus:   (Base / 2) + (proximity_to_target × ((1 - 2ε) / 50) / 2) — To strictly prevent the maximum score from exceeding 1.0, the reward budget is split 50/50 between survival and orbit proximity.
```

Episode Score Limits:
The episode score is the sum of step rewards, kept strictly inside the open interval (0, 1). To ensure standard string formatting (e.g., 3 decimal places) doesn't accidentally round boundary values to 0.000 or 1.000, we use an epsilon (ε) of 1e-3. The environment's maximum ceiling is dynamically scaled by (1 - 2ε), and the inference script floors completely failed runs by adding ε. This guarantees the grader always receives a meaningfully bounded value. A total score above 0.1 counts as success.

---

## Implemented Features

### LLM-as-Judge for Maneuver Quality

The environment ships with an **optional LLM judge** (`judge.py`) that evaluates the *reasoning quality* of each burn decision, not just whether satellites survived.

Enable it by setting `ENABLE_JUDGE=true` in your environment. The judge uses `Qwen/Qwen2.5-Coder-32B-Instruct` via the Hugging Face free serverless API and requires `HF_TOKEN` to be set.

**How it works:**
- Before each physics step, the judge receives: the observation before the action, the action taken, and the resulting observation
- It scores the maneuver from **-1.0** (harmful/wasteful) to **+1.0** (well-timed and precise), following these rules:
  - No action needed and none taken → 0.0 to 1.0
  - Unnecessary burn that wastes fuel → -1.0 to -0.5
  - Critical debris avoided cleanly → 0.8 to 1.0
  - Imminent debris missed entirely → -1.0
- The judge score scales `step_reward` via a ±20% multiplier: `reward = (reward / 1.2) × (1.0 + 0.2 × judge_score)`
- The judge's reasoning is fed back to the agent in real time through `critical_alerts`, so the LLM can course-correct its strategy mid-episode

This is especially meaningful for Task 2, where the reward function penalises over-correction but cannot distinguish *why* a burn happened.

```bash
# Enable judge in your .env
ENABLE_JUDGE=true
HF_TOKEN=hf_your_token_here
```

### Partial Observability — Radar Range Limits

The environment supports a configurable **radar horizon** that restricts what the agent can see. When `RADAR_RANGE > 0`, the `radar_debris` field in the observation only includes debris within that many position units of at least one active satellite. Debris outside the cone is invisible — but it still moves, still collides, and still cascades.

Set `RADAR_RANGE=0` (the default) to restore full observability, where all debris is always visible. This is the backwards-compatible baseline behaviour.

```bash
# Full observability (default)
RADAR_RANGE=0

# Moderate partial observability — good starting point
RADAR_RANGE=50

# Tight cone — forces predictive reasoning
RADAR_RANGE=35
```

With radar limits active, the observation includes a `radar_range` field so the agent knows its sensor horizon. An agent operating in partial observability must develop predictive reasoning — inferring the trajectory of unseen debris from orbital mechanics rather than reacting to a complete picture.

---

## Project Direction — Where This Is Headed

This is a functional baseline, but there's a lot of room to make it genuinely useful as an RL training and evaluation platform. Here's the roadmap:

### 1. GRPO Fine-Tuning with TRL + Unsloth

The reward signal here is well-structured enough to train on directly. The next step is a Colab notebook that:

1. Connects to the HF Space via WebSocket
2. Uses **Unsloth** to load a small model (Qwen3-1.7B or similar) with LoRA adapters in BF16
3. Runs rollouts through the environment and feeds rewards to **TRL's GRPO trainer**
4. Pushes checkpoints to the HF Hub

The per-step shaped reward (rather than a sparse episode-end score) makes this environment well-suited for GRPO, which benefits from variance in the reward signal across parallel rollouts. A completely dead episode and a partially successful one should look very different to the trainer — and they do.

### 2. Adversarial Debris Designer

Static debris configurations have a ceiling — once the model learns the 25-piece warmup layout well enough, it stops improving. The fix is an **adversarial scenario generator**: an LLM (Claude or similar) that inspects the agent's tracked failure modes and designs debris configurations that specifically target them.

If the agent tends to ignore debris approaching from the +Y direction, the designer spawns more of those. If the agent over-burns on Task 2, the designer creates situations where the temptation to burn is high but the debris actually misses — training the model to wait.

This creates a self-improving training loop: the better the agent gets, the harder the environment fights back.

### 3. Adaptive Curriculum

Currently difficulty is fixed: three tasks in rotation, same parameters every time. The plan is to replace this with a mastery tracker that monitors per-task success rate and adjusts the debris count, fuel budget, and rogue spawn probability in real time.

A model that's consistently scoring above `0.7` on Task 1 should be getting 35 debris pieces, not 25. One that's struggling on Task 3 shouldn't be fighting rogue spawns until its rendezvous success rate crosses a threshold.

### 4. Formation Flying Task

A fourth task type: maintain relative positions between the three satellites (e.g., keep them 20 units apart in a triangular formation). This tests multi-agent coordination and is directly relevant to real satellite constellation operations (think GPS or Starlink maintenance maneuvers).

### 5. Collision Probability (Pc) Grader

Binary collision detection is a simplification. The real-world metric is **probability of collision (Pc)** — a function of relative velocity, object sizes, and covariance uncertainty. Replacing hard collision boundaries with a Pc-weighted reward would make the environment more physically accurate and prevent the agent from learning to just barely avoid the `2.0` unit threshold by skating past debris.

### 6. Richer Physics (J2 Perturbation)

The current gravity model is point-mass Newtonian. Real LEO orbits have **J2 perturbation** from Earth's oblateness, which causes orbital planes to precess over time. Adding J2 would make the long-horizon planning in Task 3 significantly harder and more realistic — the agent would need to account for orbit drift, not just instantaneous position.

---

## Action & Observation Schema

### Action — `KesslerAction`

```json
{
  "burns": [
    { "satellite_id": 0, "delta_vx": 0.3, "delta_vy": -0.1 },
    { "satellite_id": 2, "delta_vx": 0.0, "delta_vy":  0.5 }
  ]
}
```

Each burn applies a velocity delta to one satellite, clamped to `[-1.0, 1.0]`. Fuel cost is `sqrt(dvx² + dvy²) × 5.0`. Pass `{"burns": []}` to hold position — it's free.

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
  "radar_range": 0.0,
  "critical_alerts": ["CRITICAL: Sat 1 collided with Debris 7!"],
  "done": false,
  "reward": 0.02,
  "total_score": 0.14
}
```

> **Note on `radar_range`:** When `radar_range > 0`, only debris within that distance of an active satellite appears in `radar_debris`. The field value tells the agent its own sensor horizon — debris beyond it still exists and can still collide. When `radar_range` is `0.0`, all debris is visible (full observability mode).

---

## Quick Start

### Prerequisites

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint (e.g. `https://api.openai.com/v1`) |
| `MODEL_NAME` | Model identifier (e.g. `gpt-4o`) |
| `HF_TOKEN` | Your Hugging Face / API key |
| `ENV_URL` | Running environment URL (default: `http://localhost:8000`) |
| `LOG_LEVEL` | Logging verbosity: `DEBUG`, `INFO`, or `WARNING` (default) |
| `ENABLE_JUDGE` | Set to `true` to enable LLM-as-Judge maneuver scoring (default: `false`) |
| `RADAR_RANGE` | Sensor horizon in position units; `0` = full observability (default: `0`) |

### Running Locally

**1. Start the environment server:**
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

**2. Set up your `.env`** (never commit this):
```bash
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
HF_TOKEN=hf_your_token_here
ENV_URL=http://localhost:8000
LOG_LEVEL=INFO

# Optional features
ENABLE_JUDGE=false
RADAR_RANGE=0
```

**3. Run the inference script:**
```bash
python inference.py
```

**Expected output:**
```
[START] task=task_1_easy env=kessler-env model=gpt-4o
[STEP] step=1 action={'burns':[]} reward=0.02 done=false error=null
[STEP] step=2 action={'burns':[{'satellite_id':0,'delta_vx':0.2,'delta_vy':-0.1}]} reward=0.02 done=false error=null
...
[END] success=true steps=50 score=0.847 rewards=0.02,0.02,...
```

### Using the Python Client

```python
from kessler_env import KesslerEnv, KesslerAction

with KesslerEnv(base_url="https://<your-space>.hf.space") as env:
    obs = env.reset()
    print(obs.observation.mission_objective)
    
    for step in range(50):
        # Your policy here — this just holds position
        result = env.step(KesslerAction(burns=[]))
        
        if result.observation.critical_alerts:
            print("Alerts:", result.observation.critical_alerts)
        
        if result.observation.done:
            print(f"Done — score: {result.observation.total_score:.3f}")
            break
```

### WebSocket (low-latency sessions)

```python
import websockets, json, asyncio

async def run():
    async with websockets.connect("wss://<space-url>/ws") as ws:
        await ws.send(json.dumps({"type": "reset"}))
        obs = json.loads(await ws.recv())

        action = {"burns": [{"satellite_id": 0, "delta_vx": 0.1, "delta_vy": 0.0}]}
        await ws.send(json.dumps({"type": "step", "data": action}))
        result = json.loads(await ws.recv())

        await ws.send(json.dumps({"type": "close"}))

asyncio.run(run())
```

---

## Logging

Log level is controlled by `LOG_LEVEL` in your `.env`. Logs go to **stderr** so they never interfere with `[START]`/`[STEP]`/`[END]` stdout format.

```bash
# See what's happening during a run
LOG_LEVEL=INFO python inference.py

# Full trace for debugging environment or LLM issues
LOG_LEVEL=DEBUG python inference.py 2>debug.log
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/web` | GET | Interactive web UI |
| `/docs` | GET | OpenAPI / Swagger docs |
| `/health` | GET | Container health check |
| `/reset` | POST | Reset environment, returns initial observation |
| `/step` | POST | Execute an action, returns next observation |
| `/state` | GET | Current environment state (episode ID, step count) |
| `/schema` | GET | Action and observation JSON schemas |
| `/ws` | WebSocket | Persistent low-latency session |

---

## Deployment

### HF Spaces

```bash
openenv push
# or
openenv push --repo-id my-org/kessler-env --private
```

Set `HF_TOKEN`, `API_BASE_URL`, and `MODEL_NAME` as **Space Secrets** (Settings → Variables and Secrets). They're injected as environment variables at runtime.

### Docker

```bash
docker build -t kessler_env_env:latest -f ./Dockerfile .

docker run -p 8000:8000 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o \
  -e HF_TOKEN=hf_your_token \
  -e LOG_LEVEL=INFO \
  -e ENABLE_JUDGE=false \
  -e RADAR_RANGE=0 \
  kessler_env_env:latest
```

---

## Project Structure

```
kessler_env/
├── openenv.yaml                    # OpenEnv spec manifest
├── inference.py                    # LLM agent inference script (entry point)
├── models.py                       # KesslerAction & KesslerObservation Pydantic schemas
├── client.py                       # KesslerEnv sync client (EnvClient wrapper)
├── logger.py                       # Centralised logger (LOG_LEVEL from .env)
├── judge.py                        # LLM-as-Judge maneuver evaluator (optional)
├── Dockerfile                      # Container definition
├── README.md                       # This file
├── .dockerignore
├── __init__.py
├── pyproject.toml
├── uv.lock
└── server/
    ├── __init__.py
    ├── app.py                      # FastAPI app (HTTP + WebSocket)
    ├── README.md                   # Web interface guide for human users
    └── kessler_env_environment.py  # Core physics, reward logic, episode management
```

## Inference Script Requirements

`inference.py` emits structured stdout in this format:

```
[START] task=<n> env=<benchmark> model=<model>
[STEP] step=<n> action=<json> reward=<float> done=<bool> error=<str|null>
[END] success=<bool> steps=<n> score=<float> rewards=<comma-list>
```

The script runs **3 tasks sequentially** (`task_1_easy`, `task_2_medium`, `task_3_hard`). The environment's reward ceiling is `(1 - 2ε)` and the inference script floors zero-reward runs at `ε`, so neither boundary is ever hit.