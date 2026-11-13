"""OpenAI-compatible HTTP endpoint (stdlib only).

    POST /v1/chat/completions   {"model": "...", "messages": [...]}
    GET  /v1/models

Models:
  * ``yappu-icd-linker``       – runs the clinical linker on the last user message;
                                  the assistant content is a JSON list of findings.
  * ``yappu-venba-validator``  – validates the last user message as a Nerisai Venba;
                                  the assistant content is the text report.

Any OpenAI client works, e.g.::

    from openai import OpenAI
    c = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
    c.chat.completions.create(model="yappu-icd-linker",
                              messages=[{"role": "user", "content": "Denies SOB, has HTN."}])
"""

from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from . import data_path
from .clinical import ClinicalLinker, load_lexicon
from .prosody import validate_nerisai_venba

MODELS = ("yappu-icd-linker", "yappu-venba-validator")


def _last_user_text(messages) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, list):  # content parts
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return c
    return ""


def complete(body: dict, linker: ClinicalLinker) -> dict:
    model = body.get("model", MODELS[0])
    if model not in MODELS:
        raise ValueError(f"unknown model '{model}'; available: {', '.join(MODELS)}")
    text = _last_user_text(body.get("messages"))
    if model == "yappu-icd-linker":
        found = linker.process(text)
        content = json.dumps({"findings": [f.to_dict() for f in found],
                              "allowed_codes": ClinicalLinker.allowed_codes(found)}, ensure_ascii=False)
    else:
        content = "\n".join(validate_nerisai_venba(text).lines())
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(text.split()), "completion_tokens": len(content.split()),
                  "total_tokens": len(text.split()) + len(content.split())},
    }


def make_handler(linker: ClinicalLinker):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [
                    {"id": m, "object": "model", "owned_by": "yappu-fsm"} for m in MODELS]})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._send(404, {"error": {"message": "not found"}})
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if body.get("stream"):
                    raise ValueError("streaming is not supported")
                self._send(200, complete(body, linker))
            except ValueError as e:
                self._send(400, {"error": {"message": str(e), "type": "invalid_request_error"}})

        def log_message(self, *args):  # quiet
            pass

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8000, lexicon_path: Optional[str] = None) -> ThreadingHTTPServer:
    linker = ClinicalLinker(load_lexicon(lexicon_path or data_path("icd10cm_example_lexicon.json")))
    return ThreadingHTTPServer((host, port), make_handler(linker))
