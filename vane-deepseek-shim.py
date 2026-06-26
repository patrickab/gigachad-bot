#!/usr/bin/env python3
"""Vane -> DeepSeek shim.

DeepSeek's API rejects `response_format: {type: "json_schema"}` ("This
response_format type is unavailable now"), but Vane's query planner
(Vercel AI SDK `generateObject`) always sends it. This proxy rewrites those
requests to DeepSeek's supported `{type: "json_object"}` mode and folds the
schema into a system instruction so the model still emits conforming JSON.
Everything else (including streamed answers) is passed through untouched.

ponytail: stdlib only — runs in a bare python:slim container, no pip install.
Framing is connection-close (HTTP/1.1, no content-length) so streaming SSE
just relays bytes until DeepSeek closes; no hand-rolled chunked encoding.
Auth passes through: Vane sends the Bearer key, the shim never stores it.
"""
import http.client
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "api.deepseek.com"
_STRIP = {"transfer-encoding", "connection", "content-length", "content-encoding", "keep-alive"}


def _rewrite(body: bytes) -> bytes:
    """json_schema -> json_object + schema-as-instruction. No-op otherwise."""
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return body
    rf = data.get("response_format")
    if not (isinstance(rf, dict) and rf.get("type") == "json_schema"):
        return body
    schema = (rf.get("json_schema") or {}).get("schema")
    data["response_format"] = {"type": "json_object"}
    instr = "Respond with a single valid JSON object only, no markdown fences."
    if schema is not None:
        instr += " It must conform to this JSON schema: " + json.dumps(schema)
    data.setdefault("messages", []).insert(0, {"role": "system", "content": instr})
    return json.dumps(data).encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _proxy(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = _rewrite(self.rfile.read(length)) if length else b""

        headers = {"Content-Type": "application/json", "Accept-Encoding": "identity"}
        if self.headers.get("Authorization"):
            headers["Authorization"] = self.headers["Authorization"]
        if body:
            headers["Content-Length"] = str(len(body))

        conn = http.client.HTTPSConnection(UPSTREAM, timeout=600)
        conn.request(self.command, self.path, body=body, headers=headers)
        resp = conn.getresponse()

        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in _STRIP:
                self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        while chunk := resp.read(4096):
            self.wfile.write(chunk)
            self.wfile.flush()
        conn.close()

    do_POST = do_GET = _proxy

    def log_message(self, *a) -> None:  # keep container logs quiet
        pass


def _selftest() -> None:
    # json_schema gets rewritten to json_object + a schema instruction message.
    out = json.loads(_rewrite(json.dumps({
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
    }).encode()))
    assert out["response_format"] == {"type": "json_object"}, out["response_format"]
    assert out["messages"][0]["role"] == "system" and "schema" in out["messages"][0]["content"]
    # Anything without json_schema passes through byte-for-byte.
    plain = json.dumps({"messages": [], "response_format": {"type": "json_object"}}).encode()
    assert _rewrite(plain) == plain
    assert _rewrite(b"not json") == b"not json"
    print("ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        port = int(os.environ.get("PORT", "8000"))
        ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
