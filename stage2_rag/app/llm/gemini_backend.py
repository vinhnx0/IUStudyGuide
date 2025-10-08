from __future__ import annotations

import os
from typing import Any, Dict

import google.generativeai as genai


class GeminiLLM:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.model_name = cfg["llm"]["gemini"]["model"]
        self.max_output_tokens = int(cfg["llm"]["gemini"].get("max_output_tokens", 600))
        self.temperature = float(cfg["llm"]["gemini"].get("temperature", 0.2))

        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            # Không có key -> trả thông điệp rõ ràng (để UI vẫn chạy)
            self._error = (
                "Gemini is not configured. Please set GOOGLE_API_KEY in your .env.\n\nCitations:\n-"
            )
        else:
            self._error = None
            genai.configure(api_key=key)
            self._model = genai.GenerativeModel(self.model_name)

    def generate(self, prompt: str) -> str:
        if self._error:
            return self._error
        try:
            resp = self._model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                },
                safety_settings=[
                    # nới lỏng mức mặc định để tránh block vô cớ các prompt học thuật
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ],
            )
            # resp.text là chuỗi kết quả đã kết hợp các parts
            return (resp.text or "").strip()
        except Exception as e:
            return f"(Gemini error) {e}\n\nCitations:\n-"
