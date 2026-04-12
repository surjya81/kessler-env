import os
import json
import textwrap
from openai import OpenAI

try:
    from logger import get_logger
except ImportError:
    from .logger import get_logger  # type: ignore

logger = get_logger(__name__)

class ManeuverJudge:
    def __init__(self):
        self.api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        # Free Serverless API from Hugging Face
        self.base_url = "https://router.huggingface.co/v1"
        self.model = "Qwen/Qwen2.5-Coder-32B-Instruct"
        
        self.enabled = str(os.getenv("ENABLE_JUDGE", "false")).lower() == "true"
        
        if self.enabled and not self.api_key:
            logger.warning("ENABLE_JUDGE is true, but HF_TOKEN is missing! Judge disabled.")
            self.enabled = False
            self.client = None
        elif self.enabled:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            logger.info("ManeuverJudge initialized with model: %s", self.model)
        else:
            self.client = None

    def evaluate(self, obs_before: dict, action: dict, obs_after: dict) -> tuple[float, str]:
        """
        Calls the LLM Judge to evaluate the maneuver.
        Returns:
            score (float): A value between -1.0 (terrible) and 1.0 (perfect).
            reason (str): The judge's rationale.
        """
        if not self.enabled or not self.client:
            return 0.0, "Judge disabled."

        prompt = textwrap.dedent(f"""
            You are an expert Space Traffic Control Assessor. Evaluate the AI agent's thruster action.
            
            Context Before Step:
            {json.dumps(obs_before, indent=2)}
            
            Action Taken:
            {json.dumps(action, indent=2)}
            
            Context After Step:
            {json.dumps(obs_after, indent=2)}
            
            Rules for Scoring:
            - If no action was needed (safe) and none taken: Score 0.0 to 1.0.
            - If action was taken but unnecessary (wasting fuel): Score -1.0 to -0.5.
            - If action was critical and successfully avoided debris: Score 0.8 to 1.0.
            - If agent failed to dodge imminent debris: Score -1.0.
            
            Return ONLY a valid JSON object matching this schema exactly:
            {{"reasoning": "<short sentence explaining why>", "score": <float between -1.0 and 1.0>}}
        """).strip()

        try:
            logger.debug("Judge: Requesting evaluation...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=150,
                timeout=5.0  # Keep it fast so env doesn't hang forever
            )
            
            result_text = response.choices[0].message.content or "{}"
            result_json = json.loads(result_text)
            
            score = float(result_json.get("score", 0.0))
            score = max(-1.0, min(1.0, score))  # Clamp just in case
            reason = result_json.get("reasoning", "No reason provided.")
            
            logger.debug(f"Judge output: score={score:.2f}, reason={reason}")
            return score, reason
            
        except Exception as e:
            logger.error("ManeuverJudge API failed: %s", e)
            # Default to neutral on API failure to not penalize randomly
            return 0.0, f"Judge API Error: {str(e)}"

# Singleton for performance
_JUDGE_INSTANCE = None
def get_judge() -> ManeuverJudge:
    global _JUDGE_INSTANCE
    if _JUDGE_INSTANCE is None:
        _JUDGE_INSTANCE = ManeuverJudge()
    return _JUDGE_INSTANCE