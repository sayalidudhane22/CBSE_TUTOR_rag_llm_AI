"""
sarvam_llm.py
Custom LangChain LLM wrapper for Sarvam AI API.
Sarvam AI provides Indian language-capable LLMs via an OpenAI-compatible API.
"""

import os
import requests
from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun


SARVAM_API_BASE = "https://api.sarvam.ai/v1"


class SarvamLLM(LLM):
    """
    Custom LangChain LLM wrapper for Sarvam AI.
    Sarvam AI is an Indian AI company providing multilingual LLMs.

    Usage:
        llm = SarvamLLM(api_key="your_sarvam_api_key")
        response = llm.invoke("Explain photosynthesis")
    """

    api_key: str
    model: str = "sarvam-m"          # Sarvam's flagship model
    temperature: float = 0.3
    max_tokens: int = 1024
    system_prompt: str = (
        "You are an expert CBSE Science tutor for Indian students. "
        "Explain concepts clearly with examples relevant to Indian students. "
        "Use simple language and follow NCERT curriculum strictly. "
        "When solving problems, show step-by-step working."
    )

    @property
    def _llm_type(self) -> str:
        return "sarvam"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if stop:
            payload["stop"] = stop

        try:
            response = requests.post(
                f"{SARVAM_API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.HTTPError as e:
            error_body = ""
            try:
                error_body = e.response.json().get("error", {}).get("message", str(e))
            except Exception:
                error_body = str(e)
            raise ValueError(f"Sarvam API error: {error_body}") from e

        except requests.exceptions.ConnectionError:
            raise ValueError(
                "Cannot connect to Sarvam API. Check your internet connection."
            )

        except requests.exceptions.Timeout:
            raise ValueError("Sarvam API request timed out. Try again.")

        except KeyError:
            raise ValueError(
                f"Unexpected Sarvam API response format: {response.text[:200]}"
            )

    @property
    def _identifying_params(self) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
