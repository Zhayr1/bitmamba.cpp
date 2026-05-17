#!/usr/bin/env python3
"""
Simple HTTP server for BitMamba.cpp - No external dependencies
"""

import http.server
import socketserver
import json
import subprocess
import threading
from urllib.parse import urlparse, parse_qs
import time
import os

# Default getter functions
def get_binary_path(): return os.getenv("BITMAMBA_BINARY", "../build/bitmamba")
def get_model_path(): return os.getenv("BITMAMBA_MODEL_PATH", "../bitmamba_1b.bin")
def get_default_temp(): return float(os.getenv("BITMAMBA_DEFAULT_TEMP", "0.8"))
def get_default_top_p(): return float(os.getenv("BITMAMBA_DEFAULT_TOP_P", "0.9"))
def get_default_top_k(): return int(os.getenv("BITMAMBA_DEFAULT_TOP_K", "40"))
def get_default_max_tokens(): return int(os.getenv("BITMAMBA_DEFAULT_MAX_TOKENS", "200"))
def get_default_penalty(): return float(os.getenv("BITMAMBA_DEFAULT_PENALTY", "1.15"))
def get_default_min_p(): return float(os.getenv("BITMAMBA_DEFAULT_MIN_P", "0.05"))

class BitMambaHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for BitMamba.cpp API"""

    def do_GET(self):
        """Handle GET requests"""
        parsed = urlparse(self.path)

        if parsed.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {"status": "healthy", "timestamp": int(time.time())}
                ).encode()
            )

        elif parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "message": "BitMamba.cpp Simple API Server",
                        "endpoints": {
                            "/health": "Health check",
                            "/generate": "Text generation (POST)",
                        },
                    }
                ).encode()
            )

        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode())

    def do_POST(self):
        """Handle POST requests"""
        parsed = urlparse(self.path)

        if parsed.path == "/generate":
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")

            try:
                data = json.loads(body) if body else {}
                prompt = data.get("prompt", "")
                temperature = data.get("temperature")
                if temperature is None: temperature = get_default_temp()
                max_tokens = data.get("max_tokens")
                if max_tokens is None: max_tokens = get_default_max_tokens()
                penalty = data.get("penalty")
                if penalty is None: penalty = get_default_penalty()
                min_p = data.get("min_p")
                if min_p is None: min_p = get_default_min_p()
                top_p = data.get("top_p")
                if top_p is None: top_p = get_default_top_p()
                top_k = data.get("top_k")
                if top_k is None: top_k = get_default_top_k()

                if not prompt:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({"error": "Prompt is required"}).encode()
                    )
                    return

                # Call C++ executable (paths configurable via BITMAMBA_BINARY / BITMAMBA_MODEL_PATH env vars)
                cmd = [
                    get_binary_path(),
                    get_model_path(),
                    prompt,
                    "tokenizer",
                    str(temperature),
                    str(penalty),  # penalty
                    str(min_p),  # min_p
                    str(top_p),  # top_p
                    str(top_k),  # top_k
                    str(max_tokens),
                    "clean",
                ]

                result = subprocess.run(
                    cmd, capture_output=True, text=True, errors="replace", timeout=max(30, max_tokens * 2)
                )

                if result.returncode != 0:
                    raise RuntimeError(f"Inference failed: {result.stderr}")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {"prompt": prompt, "response": result.stdout.strip()}
                    ).encode()
                )

            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode())

    def log_message(self, format, *args):
        """Override to suppress default logging"""
        pass


def run_server(port=8000):
    """Run the HTTP server"""
    with socketserver.TCPServer(("", port), BitMambaHandler) as httpd:
        print(f"🚀 BitMamba.cpp Simple API Server running on port {port}")
        print(f"🔗 http://localhost:{port}")
        print("📝 POST /generate - Text generation")
        print("📊 GET /health - Health check")
        print("\nPress Ctrl+C to stop\n")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Shutting down server...")
            httpd.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BitMamba.cpp Simple API Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--host", default="", help="Host to bind to")

    # Optional generation parameters defaults
    parser.add_argument("--temp", type=float, default=0.8, help="Default temperature")
    parser.add_argument("--max-tokens", type=int, default=200, help="Default max tokens")
    parser.add_argument("--repetition-penalty", type=float, default=1.15, help="Default repetition penalty")
    parser.add_argument("--min-p", type=float, default=0.05, help="Default min-p sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Default top-p sampling")
    parser.add_argument("--top-k", type=int, default=40, help="Default top-k sampling")

    args = parser.parse_args()
    
    import os
    os.environ["BITMAMBA_DEFAULT_TEMP"] = str(args.temp)
    os.environ["BITMAMBA_DEFAULT_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["BITMAMBA_DEFAULT_PENALTY"] = str(args.repetition_penalty)
    os.environ["BITMAMBA_DEFAULT_MIN_P"] = str(args.min_p)
    os.environ["BITMAMBA_DEFAULT_TOP_P"] = str(args.top_p)
    os.environ["BITMAMBA_DEFAULT_TOP_K"] = str(args.top_k)

    run_server(args.port)
