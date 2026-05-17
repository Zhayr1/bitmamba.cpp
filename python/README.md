# BitMamba.cpp API Server

FastAPI server with OpenAI API compatibility for BitMamba.cpp inference engine.

## Features

- ✅ **OpenAI API compatible** - Use with existing OpenAI SDKs
- ✅ **FastAPI backend** - Modern, fast Python web framework
- ✅ **Simple integration** - Works with existing C++ executable
- ✅ **Chat & Completions** - Support for both endpoints
- ✅ **Streaming support** - Real-time token streaming
- ✅ **Health checks** - Monitoring endpoints
- ✅ **CORS enabled** - Cross-origin requests allowed

## Quick Start

### 1. Install Dependencies

```bash
cd python
pip install -r requirements.txt
```

### 2. Build the C++ Executable

```bash
cd ..
cmake -B build
cmake --build build
```

### 3. Download Model Weights

```bash
# Download 1B model
wget https://huggingface.co/Zhayr1/BitMamba-2-1B/resolve/main/bitmamba_cpp/bitmamba_1b.bin

# Or download 255M model
wget https://huggingface.co/Zhayr1/BitMamba-2-0.25B/resolve/main/bitmamba_cpp/bitmamba_255m.bin

# Download tokenizer
wget https://huggingface.co/Zhayr1/BitMamba-2-1B/resolve/main/bitmamba_cpp/tokenizer.bin
```

### 4. Start the Server

```bash
cd python
python server.py --model ../bitmamba_1b.bin --host 127.0.0.1 --port 8000
```

### 5. Test the API

```bash
# Using the example client
python client.py --prompt "Hello, how are you?"

# Or using curl
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "temperature": 0.8, "max_tokens": 100}'
```

## API Endpoints

### OpenAI-Compatible Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (OpenAI compatible) |
| `/v1/completions` | POST | Text completions (OpenAI compatible) |
| `/v1/models` | GET | List available models |

### Custom Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Simple text generation |
| `/health` | GET | Health check |
| `/` | GET | API documentation |

## Usage Examples

### Using OpenAI SDK

```python
import openai

# Configure OpenAI client
client = openai.OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed"  # API key not required
)

# Chat completion
response = client.chat.completions.create(
    model="bitmamba",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    temperature=0.8,
    max_tokens=200
)

print(response.choices[0].message.content)
```

### Using cURL

```bash
# Chat completion
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bitmamba",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.8,
    "max_tokens": 200
  }'

# Text completion
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bitmamba",
    "prompt": "Once upon a time",
    "temperature": 0.8,
    "max_tokens": 200
  }'
```

### Using Python Requests

```python
import requests

# Simple generation
response = requests.post(
    "http://127.0.0.1:8000/generate",
    json={
        "prompt": "The future of AI is",
        "temperature": 0.8,
        "max_tokens": 100
    }
)
print(response.json()["response"])

# OpenAI-compatible chat
response = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    json={
        "model": "bitmamba",
        "messages": [
            {"role": "user", "content": "Explain quantum computing in simple terms"}
        ],
        "temperature": 0.7
    }
)
print(response.json()["choices"][0]["message"]["content"])
```

## Server Options

```bash
python server.py --help

Options:
  --host TEXT     Host to bind to [default: 127.0.0.1]
  --port INTEGER  Port to bind to [default: 8000]
  --model TEXT    Path to model file [default: ./bitmamba_1b.bin]
  --reload        Enable auto-reload for development
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BITMAMBA_MODEL_PATH` | Path to model file | `./bitmamba_1b.bin` |

## Performance Notes

- The server uses subprocess to call the C++ executable
- Each request spawns a new inference process
- For production use, consider implementing connection pooling
- Memory usage: ~250MB for model + Python overhead

## Deployment

### Using Docker (Example)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY python/ .
COPY build/bitmamba /usr/local/bin/
COPY tokenizer.bin /app/
COPY bitmamba_1b.bin /app/

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000", "--model", "/app/bitmamba_1b.bin"]
```

### Using Systemd Service

```ini
# /etc/systemd/system/bitmamba-api.service
[Unit]
Description=BitMamba.cpp API Server
After=network.target

[Service]
Type=simple
User=bitmamba
WorkingDirectory=/opt/bitmamba
Environment="BITMAMBA_MODEL_PATH=/opt/bitmamba/bitmamba_1b.bin"
ExecStart=/usr/bin/python3 /opt/bitmamba/python/server.py --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## Troubleshooting

### Common Issues

1. **"tokenizer.bin not found"**
   - Ensure `tokenizer.bin` is in the same directory as the C++ executable
   - Or specify path with `--tokenizer` option

2. **"model.bin not found"**
   - Check the model path with `--model` option
   - Verify file permissions

3. **Slow response times**
   - First request loads model into memory
   - Subsequent requests are faster
   - Consider keeping the server running

4. **Memory issues**
   - Each inference uses ~250MB RAM
   - Monitor memory usage with `/health` endpoint

## License

MIT License - See main project LICENSE file.