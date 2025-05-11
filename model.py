# gmi_model.py
import os
import requests
from typing import Optional, List
from dotenv import load_dotenv
from langchain.llms.base import LLM

load_dotenv()
GMI_API_KEY = os.getenv("GMI_API_KEY")
if not GMI_API_KEY:
    raise EnvironmentError("GMI_API_KEY is missing in .env file.")

class GmiLLM(LLM):
    model: str = "deepseek-ai/DeepSeek-V3-0324"
    temperature: float = 0.5
    max_tokens: int = 4096

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {GMI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        response = requests.post(
            "https://api.gmi-serving.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _llm_type(self) -> str:
        return "gmi-llm"
