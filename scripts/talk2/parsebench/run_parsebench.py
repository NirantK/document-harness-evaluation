"""Run ParseBench with the converter exactly as talk 2 uses it: LightOnOCR-3-0.8B in Ollama, grounding prompt,
pages capped at 2M pixels / 200 DPI, called through Ollama's native /api/chat with thinking off. Run with the
ParseBench checkout's interpreter (`uv run` would re-sync away the provider dependencies installed with
`uv pip install aiohttp pdf2image pypdf pillow pymupdf4llm`):

    cd <ParseBench checkout> && .venv/bin/python <this file> run lightonocr_3_ollama_talk2 --group chart

Ollama's OpenAI-compatible endpoint cannot turn thinking off, and this model then returns its whole output in
the `thinking` field with empty `content`; the native endpoint with `think: false` returns it as content, which
is the call `scripts/talk2/talk2/markdown_conversion.py` makes.
"""

import importlib

import aiohttp
from parse_bench.cli import BenchCLI
from parse_bench.extensions import register_pipeline, register_provider
from parse_bench.inference.providers.base import ProviderPermanentError, ProviderTransientError
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.product import ProductType

LightOnOcr3Provider = importlib.import_module("parse_bench.inference.providers.parse.lightonocr-3").LightOnOcr3Provider


@register_provider("lightonocr-3-ollama-native")
class LightOnOcr3OllamaNativeProvider(LightOnOcr3Provider):
    """LightOnOCR-3 provider that calls Ollama's native chat endpoint with thinking disabled."""

    async def _call_api(self, session: aiohttp.ClientSession, image_b64: str) -> str:
        payload = {
            "model": self._model, "stream": False, "think": False,
            "messages": [{"role": "user", "content": self._prompt, "images": [image_b64]}],
            "options": {"temperature": self._temperature, "num_ctx": 16384, "num_predict": self._max_tokens},
        }
        async with session.post(f"{self._server_url.rstrip('/')}/api/chat", json=payload,
                                timeout=aiohttp.ClientTimeout(total=self._timeout)) as response:
            if response.status != 200:
                error_text = await response.text()
                error = ProviderTransientError if response.status in (408, 502, 503, 504) else ProviderPermanentError
                raise error(f"HTTP {response.status}: {error_text[:200]}")
            content = (await response.json())["message"]["content"]
            if not content:
                raise ProviderPermanentError("Empty content response from API")
            return content


register_pipeline(
    PipelineSpec(
        pipeline_name="lightonocr_3_ollama_talk2",
        provider_name="lightonocr-3-ollama-native",
        product_type=ProductType.PARSE,
        config={
            "model": "maternion/LightOnOCR-3:0.8b",
            "server_url": "http://localhost:11434",
            "prompt": "grounding",
            "dpi": 200,
            "max_pixels": 2_000_000,
            "temperature": 0.1,
            "max_tokens": 8192,
            "timeout": 1800,
        },
    )
)

if __name__ == "__main__":
    import fire

    fire.Fire(BenchCLI)
