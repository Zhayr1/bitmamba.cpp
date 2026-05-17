#!/usr/bin/env python3
"""
Example client for BitMamba.cpp API server
"""

import requests
import json
import argparse
from typing import Optional


class BitMambaClient:
    """Client for BitMamba.cpp API server"""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def generate(
        self, prompt: str, **kwargs
    ) -> str:
        """Simple text generation"""
        payload = {
            "prompt": prompt,
        }
        payload.update(kwargs)
        
        response = requests.post(
            f"{self.base_url}/generate",
            json=payload,
        )
        response.raise_for_status()
        return response.json()["response"]

    def chat_completion(self, messages: list, **kwargs) -> str:
        """OpenAI-compatible chat completion"""
        data = {"model": "bitmamba", "messages": messages, **kwargs}

        response = requests.post(f"{self.base_url}/v1/chat/completions", json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

    def completion(self, prompt: str, **kwargs) -> str:
        """OpenAI-compatible text completion"""
        data = {"model": "bitmamba", "prompt": prompt, **kwargs}

        response = requests.post(f"{self.base_url}/v1/completions", json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["text"]

    def list_models(self) -> list:
        """List available models"""
        response = requests.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json()["data"]

    def health(self) -> dict:
        """Check server health"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


def main():
    parser = argparse.ArgumentParser(description="BitMamba.cpp API Client")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API server URL")
    parser.add_argument("--prompt", help="Prompt for text generation")
    parser.add_argument("--chat", action="store_true", help="Use chat completion")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty")
    parser.add_argument("--min-p", type=float, default=None, help="Min-p sampling")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument("--health", action="store_true", help="Check server health")

    args = parser.parse_args()

    client = BitMambaClient(args.url)

    if args.health:
        try:
            health = client.health()
            print(f"✅ Server is healthy: {health}")
        except Exception as e:
            print(f"❌ Server health check failed: {e}")

    elif args.list_models:
        try:
            models = client.list_models()
            print("📦 Available models:")
            for model in models:
                print(f"  - {model['id']} (owned by {model['owned_by']})")
        except Exception as e:
            print(f"❌ Failed to list models: {e}")

    elif args.prompt:
        kwargs = {}
        if args.temperature is not None: kwargs["temperature"] = args.temperature
        if args.max_tokens is not None: kwargs["max_tokens"] = args.max_tokens
        if args.repetition_penalty is not None: kwargs["penalty"] = args.repetition_penalty
        if args.min_p is not None: kwargs["min_p"] = args.min_p
        if args.top_p is not None: kwargs["top_p"] = args.top_p
        if args.top_k is not None: kwargs["top_k"] = args.top_k
        
        try:
            if args.chat:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": args.prompt},
                ]
                response = client.chat_completion(
                    messages, 
                    **kwargs
                )
                print(f"💬 Chat response:\n{response}")
            else:
                response = client.generate(
                    args.prompt,
                    **kwargs
                )
                print(f"📝 Generated text:\n{response}")
        except Exception as e:
            print(f"❌ Generation failed: {e}")

    else:
        # Interactive mode
        print("🤖 BitMamba.cpp API Client (Interactive Mode)")
        print(f"🔗 Connected to: {args.url}")
        print("Type 'quit' to exit, 'help' for commands\n")

        kwargs = {}
        if args.temperature is not None: kwargs["temperature"] = args.temperature
        if args.max_tokens is not None: kwargs["max_tokens"] = args.max_tokens
        if args.repetition_penalty is not None: kwargs["penalty"] = args.repetition_penalty
        if args.min_p is not None: kwargs["min_p"] = args.min_p
        if args.top_p is not None: kwargs["top_p"] = args.top_p
        if args.top_k is not None: kwargs["top_k"] = args.top_k

        while True:
            try:
                user_input = input("> ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    break
                elif user_input.lower() == "help":
                    print("Commands:")
                    print("  help          - Show this help")
                    print("  quit/exit/q   - Exit")
                    print("  health        - Check server health")
                    print("  models        - List available models")
                    print("  chat <prompt> - Chat completion")
                    print("  <prompt>      - Text generation")
                    continue
                elif user_input.lower() == "health":
                    health = client.health()
                    print(f"✅ {health}")
                    continue
                elif user_input.lower() == "models":
                    models = client.list_models()
                    print("📦 Models:")
                    for model in models:
                        print(f"  - {model['id']}")
                    continue
                elif user_input.lower().startswith("chat "):
                    prompt = user_input[5:].strip()
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ]
                    response = client.chat_completion(
                        messages,
                        **kwargs
                    )
                    print(f"🤖 {response}")
                else:
                    response = client.generate(
                        user_input,
                        **kwargs
                    )
                    print(f"📝 {response}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
