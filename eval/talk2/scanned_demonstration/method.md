# scanned_demonstration: two questions that need OCR

The unseen corpus plus two extracts, run separately and left out of every talk total, because no scored question rests on a page
with an empty text layer.

- MIL-STD-810H Method 516.8, pages 36 to 41 of the XL-DocBench copy. The evidence page has no text layer. Question `adubench_single_000733`.
- TO 00-25-172CL-3, Figure 7-4, from ViDoRe v3. The text layer holds only the caption. Question `industrial:239`, alias "stowage bracket".

Both extracts are US Department of Defense documents, Distribution Statement A. Model: Qwen3.5-9B, fp4, Darkbloom through
OpenRouter, seed 42. Conversion: LightOnOCR-3 0.8b in Ollama for pages the router sends to vision.
