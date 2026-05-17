#!/usr/bin/env python3
"""
FastAPI server for BitMamba.cpp with OpenAI API compatibility
"""

import asyncio
import json
import time
import subprocess
import threading
import os
import shutil
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager


# Default getter functions
def get_default_temp(): return float(os.getenv("BITMAMBA_DEFAULT_TEMP", "0.8"))
def get_default_top_p(): return float(os.getenv("BITMAMBA_DEFAULT_TOP_P", "0.9"))
def get_default_top_k(): return int(os.getenv("BITMAMBA_DEFAULT_TOP_K", "40"))
def get_default_max_tokens(): return int(os.getenv("BITMAMBA_DEFAULT_MAX_TOKENS", "200"))
def get_default_penalty(): return float(os.getenv("BITMAMBA_DEFAULT_PENALTY", "1.15"))
def get_default_min_p(): return float(os.getenv("BITMAMBA_DEFAULT_MIN_P", "0.05"))


# Models for OpenAI API compatibility
class ChatCompletionRequest(BaseModel):
    model: str = "bitmamba"
    messages: List[Dict[str, str]]
    temperature: Optional[float] = Field(default_factory=get_default_temp)
    top_p: Optional[float] = Field(default_factory=get_default_top_p)
    top_k: Optional[int] = Field(default_factory=get_default_top_k)
    max_tokens: Optional[int] = Field(default_factory=get_default_max_tokens)
    penalty: Optional[float] = Field(default_factory=get_default_penalty)
    min_p: Optional[float] = Field(default_factory=get_default_min_p)
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class CompletionRequest(BaseModel):
    model: str = "bitmamba"
    prompt: str
    temperature: Optional[float] = Field(default_factory=get_default_temp)
    top_p: Optional[float] = Field(default_factory=get_default_top_p)
    top_k: Optional[int] = Field(default_factory=get_default_top_k)
    max_tokens: Optional[int] = Field(default_factory=get_default_max_tokens)
    penalty: Optional[float] = Field(default_factory=get_default_penalty)
    min_p: Optional[float] = Field(default_factory=get_default_min_p)
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "bitmamba.cpp"


class BitMambaInference:
    """Wrapper for BitMamba.cpp inference using subprocess"""

    def __init__(self, model_path: str, tokenizer_path: str = "tokenizer.bin"):
        # Convert to absolute paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)

        self.model_path = os.path.abspath(model_path)
        self.executor = ThreadPoolExecutor(max_workers=1)  # Single inference at a time

        # Paths for execution
        self.build_dir = os.path.join(project_root, "build")
        self.executable_path = os.path.join(self.build_dir, "bitmamba")

        # Find tokenizer
        possible_tokenizer_locations = [
            os.path.join(self.build_dir, "tokenizer.bin"),
            os.path.join(project_root, "tokenizer.bin"),
            os.path.join(script_dir, "tokenizer.bin"),
            os.path.abspath(tokenizer_path),
        ]

        self.tokenizer_path = None
        for loc in possible_tokenizer_locations:
            if os.path.exists(loc):
                self.tokenizer_path = loc
                break

        if not self.tokenizer_path:
            raise FileNotFoundError(
                f"Tokenizer not found. Searched in: {possible_tokenizer_locations}"
            )

        # If tokenizer is not in build directory, copy it there
        target_tokenizer = os.path.join(self.build_dir, "tokenizer.bin")
        if self.tokenizer_path != target_tokenizer:
            try:
                shutil.copy2(self.tokenizer_path, target_tokenizer)
                print(f"📋 Copied tokenizer to {target_tokenizer}")
                self.tokenizer_path = target_tokenizer
            except Exception as e:
                print(f"⚠️  Could not copy tokenizer: {e}")
                # Continue with original path, but inference may fail

        # Validate paths
        if not os.path.exists(self.executable_path):
            raise FileNotFoundError(f"Executable not found: {self.executable_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the C++ executable"""
        temperature = kwargs.get("temperature")
        if temperature is None: temperature = get_default_temp()
        
        top_p = kwargs.get("top_p")
        if top_p is None: top_p = get_default_top_p()
        
        top_k = kwargs.get("top_k")
        if top_k is None: top_k = get_default_top_k()
        
        max_tokens = kwargs.get("max_tokens")
        if max_tokens is None: max_tokens = get_default_max_tokens()
        
        min_p = kwargs.get("min_p")
        if min_p is None: min_p = get_default_min_p()
        
        penalty = kwargs.get("penalty")
        if penalty is None: penalty = get_default_penalty()

        # Build command
        cmd = [
            self.executable_path,
            self.model_path,
            prompt,
            "tokenizer",
            str(temperature),
            str(penalty),
            str(min_p),
            str(top_p),
            str(top_k),
            str(max_tokens),
            "clean",  # Clean output mode
        ]

        print(f"🚀 Running command: {' '.join(cmd[:3])}... (cwd={self.build_dir})")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                errors="replace",
                cwd=self.build_dir,
                timeout=max(30, max_tokens * 2),  # Timeout based on max tokens with min 30s
            )

            if result.returncode != 0:
                print(f"❌ Command failed: {result.stderr}")
                raise RuntimeError(f"Inference failed: {result.stderr}")

            output = result.stdout.strip()

            print(f"✅ Inference succeeded: {len(output)} chars")
            return output

        except subprocess.TimeoutExpired:
            print("⏰ Inference timeout")
            raise RuntimeError("Inference timeout")
        except Exception as e:
            print(f"💥 Unexpected error: {e}")
            import traceback

            traceback.print_exc()
            raise RuntimeError(f"Inference error: {str(e)}")

    def generate_async(self, prompt: str, **kwargs):
        """Async wrapper for generation"""
        return self.executor.submit(self.generate, prompt, **kwargs)


# Global inference engine
inference_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup
    global inference_engine
    import os

    model_path = os.getenv("BITMAMBA_MODEL_PATH", "./bitmamba_1b.bin")
    inference_engine = BitMambaInference(model_path)
    print(f"✅ BitMamba.cpp API server started with model: {model_path}")

    yield

    # Shutdown
    print("👋 Shutting down BitMamba.cpp API server...")
    if inference_engine:
        inference_engine.executor.shutdown(wait=True)


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="BitMamba.cpp API",
    description="OpenAI-compatible API for BitMamba.cpp inference engine",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "BitMamba.cpp API Server",
        "endpoints": {
            "/v1/models": "List available models",
            "/v1/chat/completions": "Chat completions (OpenAI compatible)",
            "/v1/completions": "Text completions (OpenAI compatible)",
            "/health": "Health check",
            "/generate": "Simple text generation",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": int(time.time())}


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "bitmamba",
                "object": "model",
                "created": 1704067200,  # Fixed timestamp
                "owned_by": "bitmamba.cpp",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Chat completions endpoint (OpenAI compatible)"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    # Convert messages to prompt
    prompt = ""
    for msg in request.messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            prompt += f"System: {content}\n"
        elif role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"

    prompt += "Assistant: "

    # Generate response
    start_time = time.time()

    try:
        if request.stream:
            # Streaming response (simplified)
            async def stream_response():
                # For simplicity, we generate the full response first
                response = inference_engine.generate(
                    prompt,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    max_tokens=request.max_tokens,
                    penalty=request.penalty,
                    min_p=request.min_p,
                )

                # Simulate streaming by sending chunks
                for i in range(0, len(response), 10):
                    chunk = response[i : i + 10]
                    yield f"data: {
                        json.dumps(
                            {
                                'id': f'chatcmpl-{int(time.time())}',
                                'object': 'chat.completion.chunk',
                                'created': int(time.time()),
                                'model': request.model,
                                'choices': [
                                    {
                                        'index': 0,
                                        'delta': {'content': chunk},
                                        'finish_reason': None
                                        if i + 10 < len(response)
                                        else 'stop',
                                    }
                                ],
                            }
                        )
                    }\n\n"

                yield "data: [DONE]\n\n"

            from fastapi.responses import StreamingResponse

            return StreamingResponse(stream_response(), media_type="text/event-stream")

        else:
            # Non-streaming response
            response_text = inference_engine.generate(
                prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
                penalty=request.penalty,
                min_p=request.min_p,
            )

            generation_time = time.time() - start_time

            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt.split()),  # Approximate
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(prompt.split()) + len(response_text.split()),
                },
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Text completions endpoint (OpenAI compatible)"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    start_time = time.time()

    try:
        if request.stream:
            # Streaming response
            async def stream_response():
                response = inference_engine.generate(
                    request.prompt,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    max_tokens=request.max_tokens,
                    penalty=request.penalty,
                    min_p=request.min_p,
                )

                for i in range(0, len(response), 10):
                    chunk = response[i : i + 10]
                    yield f"data: {
                        json.dumps(
                            {
                                'id': f'cmpl-{int(time.time())}',
                                'object': 'text_completion.chunk',
                                'created': int(time.time()),
                                'model': request.model,
                                'choices': [
                                    {
                                        'index': 0,
                                        'text': chunk,
                                        'finish_reason': None
                                        if i + 10 < len(response)
                                        else 'stop',
                                    }
                                ],
                            }
                        )
                    }\n\n"

                yield "data: [DONE]\n\n"

            from fastapi.responses import StreamingResponse

            return StreamingResponse(stream_response(), media_type="text/event-stream")

        else:
            # Non-streaming response
            response_text = inference_engine.generate(
                request.prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
                penalty=request.penalty,
                min_p=request.min_p,
            )

            return {
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {"index": 0, "text": response_text, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": len(request.prompt.split()),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(request.prompt.split())
                    + len(response_text.split()),
                },
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate")
async def generate_text(
    prompt: str = Body(..., embed=True),
    temperature: Optional[float] = Body(None, embed=True),
    max_tokens: Optional[int] = Body(None, embed=True),
    penalty: Optional[float] = Body(None, embed=True),
    min_p: Optional[float] = Body(None, embed=True),
    top_p: Optional[float] = Body(None, embed=True),
    top_k: Optional[int] = Body(None, embed=True),
):
    """Simple text generation endpoint"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    temperature = temperature if temperature is not None else get_default_temp()
    max_tokens = max_tokens if max_tokens is not None else get_default_max_tokens()
    penalty = penalty if penalty is not None else get_default_penalty()
    min_p = min_p if min_p is not None else get_default_min_p()
    top_p = top_p if top_p is not None else get_default_top_p()
    top_k = top_k if top_k is not None else get_default_top_k()

    print(
        f"🔧 /generate called: prompt='{prompt[:50]}...', temp={temperature}, max_tokens={max_tokens}"
    )

    try:
        response = inference_engine.generate(
            prompt, 
            temperature=temperature, 
            max_tokens=max_tokens,
            penalty=penalty,
            min_p=min_p,
            top_p=top_p,
            top_k=top_k
        )
        return {"prompt": prompt, "response": response}
    except Exception as e:
        print(f"❌ Error in /generate: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BitMamba.cpp API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--model", default="./bitmamba_1b.bin", help="Path to model file"
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Optional generation parameters defaults
    parser.add_argument("--temp", type=float, default=0.8, help="Default temperature")
    parser.add_argument("--max-tokens", type=int, default=200, help="Default max tokens")
    parser.add_argument("--repetition-penalty", type=float, default=1.15, help="Default repetition penalty")
    parser.add_argument("--min-p", type=float, default=0.05, help="Default min-p sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Default top-p sampling")
    parser.add_argument("--top-k", type=int, default=40, help="Default top-k sampling")

    args = parser.parse_args()

    # Set model path environment variable
    import os

    os.environ["BITMAMBA_MODEL_PATH"] = args.model
    os.environ["BITMAMBA_DEFAULT_TEMP"] = str(args.temp)
    os.environ["BITMAMBA_DEFAULT_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["BITMAMBA_DEFAULT_PENALTY"] = str(args.repetition_penalty)
    os.environ["BITMAMBA_DEFAULT_MIN_P"] = str(args.min_p)
    os.environ["BITMAMBA_DEFAULT_TOP_P"] = str(args.top_p)
    os.environ["BITMAMBA_DEFAULT_TOP_K"] = str(args.top_k)

    print(f"🚀 Starting BitMamba.cpp API server on {args.host}:{args.port}")
    print(f"📦 Model: {args.model}")
    print(f"🔗 OpenAI-compatible endpoints available at:")
    print(f"   - http://{args.host}:{args.port}/v1/chat/completions")
    print(f"   - http://{args.host}:{args.port}/v1/completions")

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
