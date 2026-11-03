"""OpenAICompatLLM against a minimal OpenAI-style HTTP server (stands in for
llama-server): request format, usage accounting, cache hits."""
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI

from conftest import GOOD_TEST, script
from testagent.llm import LLMConfig, OpenAICompatLLM
from testagent.schemas import IssueAnalysis


def serve(replies, seen):
    app = FastAPI()

    @app.post("/v1/chat/completions")
    def chat(body: dict):
        user = body["messages"][1]["content"]
        task = user.split("\n", 1)[0].removeprefix("TASK: ")
        seen.append((body["model"], body["temperature"], task))
        text = "Here you go:\n```json\n" + replies[task][0] + "\n```"
        return {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168}}
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    return server, port


def test_openai_compatible_client_with_cache(tmp_path):
    seen = []
    server, port = serve(script([GOOD_TEST]), seen)
    try:
        cfg = LLMConfig(model="qwen-test", base_url=f"http://127.0.0.1:{port}/v1", cache_dir=str(tmp_path))
        llm = OpenAICompatLLM(cfg)
        a = llm.structured("analyze_issue", "issue text", IssueAnalysis)
        assert a.requirements[0].id == "R1"
        assert llm.calls[0].prompt_tokens == 123 and not llm.calls[0].cached
        b = OpenAICompatLLM(cfg).structured("analyze_issue", "issue text", IssueAnalysis)
        assert b == a and len(seen) == 1                       # second answer came from the cache
        assert seen[0] == ("qwen-test", 0.0, "analyze_issue")
    finally:
        server.should_exit = True
