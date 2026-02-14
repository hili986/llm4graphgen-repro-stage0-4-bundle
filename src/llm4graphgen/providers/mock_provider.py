"""离线 Mock Provider。"""

from __future__ import annotations

from llm4graphgen.providers.base import BaseProvider


DEFAULT_MOCK_OUTPUT = "(4, [(0,1), (1,2), (2,3), (0,1),])"


class MockProvider(BaseProvider):
    """用于 Stage1 冒烟链路的固定输出 Provider。"""

    name = "mock"

    def __init__(self, fixed_output: str | None = None) -> None:
        self._fixed_output = fixed_output or DEFAULT_MOCK_OUTPUT

    def generate(self, prompt: str, model: str, temperature: float) -> str:
        _ = (prompt, model, temperature)
        return self._fixed_output
