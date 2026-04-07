"""
Inference Script for Kessler-Env
Uses WebSocket (not HTTP) for stateful sessions — OpenEnv's HTTP /reset and /step
endpoints create a brand-new environment instance per request and immediately
discard it, so state never persists. The /ws WebSocket endpoint is the only path
that keeps a single environment instance alive across reset → step → step → ...
"""
import os
import json
import textwrap
import asyncio
import websockets
from typing import List, Optional
from openai import OpenAI

# --- Hackathon Required Envs ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
TASK_NAME = os.getenv("TASK_NAME", "kessler-survival")
BENCHMARK = os.getenv("BENCHMARK", "kessler-env")
MAX_STEPS = 50

# Max possible reward across the full episode.
# Each step: all 3 sats alive → (3/3) × (1/50) = 0.02. Over 50 steps → 1.0 total.
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.1  # normalised score in [0, 1]

# WebSocket URL — note ws:// scheme, same host/port as the HTTP server
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
WS_URL = ENV_URL.replace("http://", "ws://").replace("https://", "wss://").rstrip("/") + "/ws"

SYSTEM_PROMPT = textwrap.dedent("""
    You are an autonomous orbital traffic controller.
    Your goal is to avoid space debris collisions while conserving fuel.
    Thruster max delta is 1.0 or -1.0.

    Return EXACTLY ONE JSON block matching this schema:
    {"burns":[{"satellite_id": int, "delta_vx": float, "delta_vy": float}]}

    If no maneuvers are needed, return: {"burns":[]}
""").strip()


# ---------------------------------------------------------------------------
# Logging helpers (hackathon stdout spec)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# WebSocket helpers
# ---------------------------------------------------------------------------

async def ws_send_recv(ws, payload: dict) -> dict:
    """Send a JSON message and receive the next JSON response."""
    await ws.send(json.dumps(payload))
    raw = await ws.recv()
    return json.loads(raw)


def parse_obs(response: dict) -> dict:
    """
    Extract the observation dict from a WSObservationResponse.
    OpenEnv wraps the full StepResponse/ResetResponse inside a 'data' key:
        {"type": "observation", "data": {"observation": {...}, "reward": 0.0, "done": false}}
    """
    return response.get("data", response)


# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------

async def run(client: OpenAI) -> tuple[bool, int, float, List[float]]:
    rewards: List[float] = []
    steps_taken = 0

    async with websockets.connect(WS_URL) as ws:

        # 1. Reset — initialises the persistent env instance for this WS session
        reset_response = await ws_send_recv(ws, {"type": "reset"})
        step_data = parse_obs(reset_response)
        observation = step_data.get("observation", step_data)

        for step in range(1, MAX_STEPS + 1):

            # 2. Build prompt
            user_content = textwrap.dedent(f"""
            Step: {step}
            Current Telemetry: {json.dumps(observation, indent=2)}
            Reply with exactly one JSON object for thruster burns.
            """).strip()

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            error_msg = None
            action_str = "{'burns':[]}"
            action_dict: dict = {"burns": []}
            reward = 0.0
            done = False

            # 3. Call LLM
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                response_text = completion.choices[0].message.content or "{}"
                action_dict = json.loads(response_text)
                action_str = json.dumps(action_dict).replace('"', "'")
            except Exception as exc:
                error_msg = f"LLM Error: {str(exc)}"

            # 4. Step environment over WebSocket
            if not error_msg:
                try:
                    step_response = await ws_send_recv(ws, {"type": "step", "data": action_dict})

                    if step_response.get("type") == "error":
                        error_msg = f"Env Error: {step_response.get('data', {}).get('message', 'unknown')}"
                        done = True
                    else:
                        step_data = parse_obs(step_response)
                        reward = float(step_data.get("reward", 0.0))
                        done = bool(step_data.get("done", False))
                        observation = step_data.get("observation", step_data)
                except Exception as exc:
                    error_msg = f"Env Error: {str(exc)}"
                    done = True

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        # Gracefully close the WebSocket session
        try:
            await ws_send_recv(ws, {"type": "close"})
        except Exception:
            pass  # server may already have closed on done=true

    score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
    score = min(max(score, 0.0), 1.0)
    success = score >= SUCCESS_SCORE_THRESHOLD
    return success, steps_taken, score, rewards


def main() -> None:
    if not API_KEY:
        raise ValueError("HF_TOKEN or API_KEY environment variable is missing!")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        success, steps_taken, score, rewards = asyncio.run(run(client))
    except Exception as fatal_e:
        log_step(step=0, action="startup", reward=0.0, done=True, error=str(fatal_e))
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()