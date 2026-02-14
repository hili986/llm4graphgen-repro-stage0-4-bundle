"""Provider 抽象层。"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """统一的 LLM 调用接口。"""

    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, model: str, temperature: float) -> str:
        """返回模型原始文本输出。"""
        raise NotImplementedError
