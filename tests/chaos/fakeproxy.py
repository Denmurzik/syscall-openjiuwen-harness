"""In-process fake OpenAI-compatible proxy with scripted hostile behaviours (stdlib only)."""
import json
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NORMAL = "normal"


def canned(body: dict) -> str:
    """Valid answer chosen by prompt content: JSON specs, programs, arbitration digits."""
    msgs = body.get("messages") or []
    system = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "")
    system = system if isinstance(system, str) else ""
    if "arithmetic word problem" in system:
        return '{"start": 10, "steps": ["x+1"]}'
    if "CSV file" in system or "knowledge base" in system:
        return '{"impossible": true}'
    if "Write a Python 3 program" in system:
        return "print(42)"
    if "digit" in system.lower() or "choose" in system.lower():
        return "1"
    return "DONE"


class Proxy:
    """behavior: callable(n, body) -> action, n is the 1-based request counter.

    Actions: normal, 429, 429_retry_after, 500, 502, 503, 402, 403, 400_structured (400 only when
    the request carries response_format/extra_body, else normal), timeout, hang, reset, empty,
    malformed, no_usage, huge_usage.
    """

    def __init__(self, behavior=None, sleep_s: float = 30.0):
        self.behavior = behavior or (lambda n, body: NORMAL)
        self.sleep_s = sleep_s
        self.requests = 0
        self.actions: list = []
        self.bodies: list = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        proxy = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    raw = self.rfile.read(n) if n else b""
                    try:
                        body = json.loads(raw or b"{}")
                    except ValueError:
                        body = {}
                    with proxy._lock:
                        proxy.requests += 1
                        idx = proxy.requests
                    try:
                        action = proxy.behavior(idx, body)
                    except Exception:
                        action = NORMAL
                    with proxy._lock:
                        proxy.actions.append(action)
                        proxy.bodies.append(body)
                    proxy._respond(self, action, body)
                except (ConnectionError, OSError):
                    pass

            def do_GET(self):
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        class S(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True
            request_queue_size = 64

        self.server = S(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.1}, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self._stop.set()
        try:
            self.server.shutdown()
        except Exception:
            pass
        try:
            self.server.server_close()
        except Exception:
            pass
        self.thread.join(5)

    def count(self, *actions) -> int:
        with self._lock:
            return sum(1 for a in self.actions if a in actions)

    # ------------------------------------------------------------------
    @staticmethod
    def _send(h, status, payload: bytes, headers=None, ctype="application/json"):
        h.send_response(status)
        h.send_header("Content-Type", ctype)
        h.send_header("Content-Length", str(len(payload)))
        for k, v in (headers or {}).items():
            h.send_header(k, v)
        h.end_headers()
        h.wfile.write(payload)
        h.wfile.flush()

    def _ok(self, h, body, usage="default"):
        msg = {"id": "chatcmpl-x", "object": "chat.completion", "created": int(time.time()),
               "model": body.get("model", "m"),
               "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": canned(body)}}]}
        if usage == "default":
            msg["usage"] = {"prompt_tokens": 300, "completion_tokens": 20, "total_tokens": 320}
        elif usage == "huge":
            msg["usage"] = {"prompt_tokens": 10 ** 12, "completion_tokens": 10 ** 12,
                            "total_tokens": 2 * 10 ** 12}
        self._send(h, 200, json.dumps(msg).encode())

    def _respond(self, h, action, body):
        def err(code, text):
            return json.dumps({"error": {"message": text, "type": "x", "code": code}}).encode()

        if action == NORMAL:
            self._ok(h, body)
        elif action == "no_usage":
            self._ok(h, body, usage=None)
        elif action == "huge_usage":
            self._ok(h, body, usage="huge")
        elif action == "429":
            self._send(h, 429, err(429, "rate limited"))
        elif action == "429_retry_after":
            self._send(h, 429, err(429, "rate limited"), {"Retry-After": "1"})
        elif action in ("500", "502", "503"):
            self._send(h, int(action), err(int(action), "upstream"))
        elif action in ("402", "403"):
            self._send(h, int(action), err(int(action), "budget exhausted"))
        elif action == "400_structured":
            if "response_format" in body or "extra_body" in body:
                self._send(h, 400, err(400, "response_format not supported"))
            else:
                self._ok(h, body)
        elif action in ("timeout", "hang"):
            self._stop.wait(self.sleep_s)
        elif action == "reset":
            try:
                h.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            except OSError:
                pass
            h.close_connection = True
            h.connection.close()
        elif action == "empty":
            self._send(h, 200, b"")
        elif action == "malformed":
            self._send(h, 200, b'{"choices": [{"message": {"content": "x"')
        else:
            raise ValueError(action)


def exhausted_after(k: int, code: str = "402", base: str = NORMAL):
    return lambda n, body: base if n <= k else code
