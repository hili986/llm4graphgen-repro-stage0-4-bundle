"""OpenAI Provider（通过环境变量读取 API Key）。"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

from llm4graphgen.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """使用 OpenAI Responses API 的最小实现。"""

    name = "openai"

    def __init__(
        self,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ) -> None:
        self._api_key = os.getenv(api_key_env)
        if not self._api_key:
            raise ValueError(f"未检测到环境变量 {api_key_env}，无法使用 OpenAIProvider。")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def generate(self, prompt: str, model: str, temperature: float) -> str:
        payload = {
            "model": model,
            "input": prompt,
            "temperature": temperature,
        }
        url = f"{self._base_url}/responses"
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

        data: dict[str, Any] = json.loads(body)
        output_text = _extract_output_text(data)
        if not output_text:
            raise RuntimeError("OpenAI 返回内容中未找到可解析文本。")
        return output_text


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
