"""OpenAI Provider — 支持 Chat Completions API 和 Responses API。"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

from llm4graphgen.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI LLM 调用实现。

    支持两种 API 模式：
    - Chat Completions（默认，兼容性更好）
    - Responses API（新版）
    """

    name = "openai"

    def __init__(
        self,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        api_mode: str = "chat",  # "chat" 或 "responses"
    ) -> None:
        self._api_key = os.getenv(api_key_env)
        if not self._api_key:
            raise ValueError(f"未检测到环境变量 {api_key_env}，无法使用 OpenAIProvider。")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._api_mode = api_mode

    def generate(self, prompt: str, model: str, temperature: float) -> str:
        if self._api_mode == "chat":
            return self._generate_chat(prompt, model, temperature)
        return self._generate_responses(prompt, model, temperature)

    def _generate_chat(self, prompt: str, model: str, temperature: float) -> str:
        """使用 Chat Completions API。"""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 4096,
        }
        url = f"{self._base_url}/chat/completions"
        data = self._post(url, payload)

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("OpenAI Chat API 返回空 choices。")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not content.strip():
            raise RuntimeError("OpenAI Chat API 返回空内容。")
        return content

    def _generate_responses(self, prompt: str, model: str, temperature: float) -> str:
        """使用 Responses API。"""
        payload = {
            "model": model,
            "input": prompt,
            "temperature": temperature,
        }
        url = f"{self._base_url}/responses"
        data = self._post(url, payload)

        output_text = _extract_output_text(data)
        if not output_text:
            raise RuntimeError("OpenAI Responses API 返回内容中未找到可解析文本。")
        return output_text

    def _post(self, url: str, payload: dict) -> dict[str, Any]:
        req = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI 请求失败（HTTP {exc.code}）：{detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenAI 网络请求失败：{exc.reason}") from exc

        return json.loads(body)


def _extract_output_text(data: dict[str, Any]) -> str:
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct

    output = data.get("output")
    if isinstance(output, list):
        texts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for chunk in content:
                if not isinstance(chunk, dict):
                    continue
                text = chunk.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text)
        return "\n".join(texts).strip()
    return ""
