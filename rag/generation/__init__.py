"""Generation package for LLM synthesis and prompts."""

from .generator import get_generator, GeminiGenerator, generate_answer
from .prompts import SYSTEM_PROMPT

__all__ = ["get_generator", "GeminiGenerator", "generate_answer", "SYSTEM_PROMPT"]
