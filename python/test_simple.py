#!/usr/bin/env python3
"""
Simple test for BitMamba.cpp API server
"""

import subprocess
import time
import requests
import json


def test_server():
    print("🧪 Testing BitMamba.cpp API server...")

    # Start server in background
    print("🚀 Starting server...")
    server_process = subprocess.Popen(
        [
            "python",
            "server.py",
            "--model",
            "../bitmamba_1b.bin",
            "--host",
            "127.0.0.1",
            "--port",
            "8001",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    time.sleep(3)

    try:
        # Test health endpoint
        print("📊 Testing health endpoint...")
        response = requests.get("http://127.0.0.1:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False

        # Test models endpoint
        print("📦 Testing models endpoint...")
        response = requests.get("http://127.0.0.1:8001/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Models endpoint OK: {len(models.get('data', []))} models")
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False

        # Test generate endpoint (simple)
        print("📝 Testing generate endpoint...")
        response = requests.post(
            "http://127.0.0.1:8001/generate",
            json={"prompt": "Hello", "max_tokens": 10},
            timeout=10,
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Generate endpoint OK: {len(result.get('response', ''))} chars")
        else:
            print(f"❌ Generate endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

        print("🎉 All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

    finally:
        # Kill server
        print("🛑 Stopping server...")
        server_process.terminate()
        server_process.wait(timeout=5)


if __name__ == "__main__":
    success = test_server()
    exit(0 if success else 1)
