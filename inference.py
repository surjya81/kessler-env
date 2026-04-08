"""
Inference Script for Kessler-Env
Uses WebSocket for stateful sessions. Runs 3 separate tasks to fulfill grader requirements.
"""
"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method
- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
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
BENCHMARK = os.getenv("BENCHMARK", "kessler-env")
MAX_STEPS = 50

MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.1 

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
# Logging helpers
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
# WebSocket helpers
# ---------------------------------------------------------------------------
async def ws_send_recv(ws, payload: dict) -> dict:
    await ws.send(json.dumps(payload))
    raw = await ws.recv()
    return json.loads(raw)

def parse_obs(response: dict) -> dict:
    return response.get("data", response)

# ---------------------------------------------------------------------------
# Main async loop (Runs for a single task episode)
# ---------------------------------------------------------------------------
async def run_episode(client: OpenAI, task_name: str) -> tuple[bool, int, float, List[float]]:
    rewards: List[float] =[]
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with websockets.connect(WS_URL) as ws:
            # 1. Reset
            reset_response = await ws_send_recv(ws, {"type": "reset"})
            step_data = parse_obs(reset_response)
            observation = step_data.get("observation", step_data)

            for step in range(1, MAX_STEPS + 1):
                # Pull the objective from the observation dict
                mission = observation.get("mission_objective", "Survive.")

                user_content = textwrap.dedent(f"""
                Step: {step} | Task: {task_name}
                Current Telemetry: {json.dumps(observation, indent=2)}
                Reply with exactly one JSON object for thruster burns.
                """).strip()

                messages =[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]

                error_msg = None
                action_str = "{'burns':[]}"
                action_dict: dict = {"burns":[]}
                reward = 0.0
                done = False

                # 2. Call LLM
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

                # 3. Step env
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

            try:
                await ws_send_recv(ws, {"type": "close"})
            except Exception:
                pass  

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as fatal_e:
        log_step(step=0, action="startup", reward=0.0, done=True, error=str(fatal_e))
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        
    return success, steps_taken, score, rewards


def main() -> None:
    if not API_KEY:
        raise ValueError("HF_TOKEN or API_KEY environment variable is missing!")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Run the 3 Hackathon Tasks in sequence
    tasks =["task_1_easy", "task_2_medium", "task_3_hard"]
    
    for task in tasks:
        asyncio.run(run_episode(client, task))


if __name__ == "__main__":
    main()