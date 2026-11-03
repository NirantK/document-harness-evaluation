"""Configurations `rag` and `markdown_rag`: retrieve top-5 chunks, then one answer call."""

import time

from talk2.answer_format import ANSWER_INSTRUCTIONS, parse_answer
from talk2.model_client import chat

SYSTEM_PROMPT = ("You answer questions about a document collection using only the retrieved passages below.\n"
                 + ANSWER_INSTRUCTIONS)


def answer_question(question: str, retriever, model: str, backend: str, think: bool = False) -> dict:
    started = time.time()
    nodes = retriever.retrieve(question)
    passages = "\n\n".join(
        f"[{index}] file: {node.node.metadata['file_path']}\n{node.node.get_content()}"
        for index, node in enumerate(nodes, 1)
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Retrieved passages:\n\n{passages}\n\nQuestion: {question}"}]
    turn = chat(messages, model=model, backend=backend, think=think)
    return {
        "response": parse_answer(turn.content),
        "raw_reply": turn.content,
        "retrieved": [{"file": n.node.metadata["file_path"], "score": round(float(n.score), 4),
                       "text": n.node.get_content()} for n in nodes],
        "files_touched": sorted({n.node.metadata["file_path"] for n in nodes}),
        "prompt_tokens": turn.prompt_tokens, "completion_tokens": turn.completion_tokens,
        "model_calls": 1, "provider": turn.provider, "wall_seconds": round(time.time() - started, 1),
    }
