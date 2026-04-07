"""
Strict Inference Script for Kessler-Env
Complies fully with Hugging Face / OpenEnv hackathon stdout guidelines.
"""
import os
import json
import textwrap
import requests
from typing import List, Optional
from openai import OpenAI

# --- Hackathon Required Envs ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
TASK_NAME = os.getenv("TASK_NAME", "kessler-survival")
BENCHMARK = os.getenv("BENCHMARK", "kessler-env")
MAX_STEPS = 50

# URL points to the OpenEnv API (local or Hugging Face Space)
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

SYSTEM_PROMPT = textwrap.dedent("""
    You are an autonomous orbital traffic controller.
    Your goal is to avoid space debris collisions while conserving fuel. 
    Thruster max delta is 1.0 or -1.0. 
    
    Return EXACTLY ONE JSON block matching this schema:
    {"burns":[{"satellite_id": int, "delta_vx": float, "delta_vy": float}]}
    
    If no maneuvers are needed, return: {"burns":[]}
""").strip()

class RemoteEnvironment:
    """
    A lightweight, robust HTTP client drop-in replacement for OpenEnv.
    Bypasses library version mismatches by talking directly to the FastAPI server.
    """
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self) -> dict:
        response = requests.post(f"{self.base_url}/reset")
        response.raise_for_status()
        return response.json()
    
    def step(self, action: dict) -> dict:
        # OpenEnv strictly requires the payload to be wrapped in an "action" key
        payload = {"action": action}
        response = requests.post(f"{self.base_url}/step", json=payload)
        response.raise_for_status()
        return response.json()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

def main() -> None:
    if not API_KEY:
        raise ValueError("HF_TOKEN or API_KEY environment variable is missing!")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = RemoteEnvironment(ENV_URL)

    rewards: List[float] =[]
    steps_taken = 0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Standard HTTP reset call
        obs_response = env.reset()
        # OpenEnv typically wraps the raw observation in an 'observation' key
        observation = obs_response.get('observation', obs_response)

        for step in range(1, MAX_STEPS + 1):
            
            # 1. Build Prompt from Telemetry
            user_content = textwrap.dedent(f"""
            Step: {step}
            Current Telemetry: {json.dumps(observation, indent=2)}
            Reply with exactly one JSON object for thruster burns.
            """).strip()

            messages =[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            error_msg = None
            action_str = "{'burns':[]}"
            action_dict = {"burns":[]}
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
                action_str = json.dumps(action_dict).replace('"', "'") # Format for single-line log
            except Exception as exc:
                error_msg = f"LLM Error: {str(exc)}"

            # 3. Step Environment via HTTP
            if not error_msg:
                try:
                    step_response = env.step(action_dict)
                    
                    reward = float(step_response.get('reward', 0.0))
                    done = bool(step_response.get('done', False))
                    observation = step_response.get('observation', step_response)
                except Exception as exc:
                    error_msg = f"Env Error: {str(exc)}"
                    reward = 0.0
                    done = True

            # 4. Log Step
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                # Standard hackathon grading condition
                success = sum(rewards) > 0.5 
                break

        else:
            success = sum(rewards) > 0.5

    except Exception as fatal_e:
        # Catch connection failures at startup
        log_step(step=0, action="startup", reward=0.0, done=True, error=str(fatal_e))
    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

if __name__ == "__main__":
    main()