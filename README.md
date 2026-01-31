# BitMamba.cpp Library

This library is designed for efficient inference using **BitMamba2** models with **255M** and **1B** parameters. It implements support for quantization and BitNet-optimized architectures.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/Paper-Zenodo-00649C.svg)](https://doi.org/10.5281/zenodo.18394665)
[![Hugging Face 1B](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-1B%20Model-FFD21E)](https://huggingface.co/Zhayr1/BitMamba-2-1B)
[![Hugging Face 255M](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-255M%20Model-FFD21E)](https://huggingface.co/Zhayr1/BitMamba-2-0.25B)

## Requirements and compatibility

⚠️ Hardware Requirements: This C++ implementation utilizes AVX2 SIMD instructions for high-performance inference on x86 CPUs (Intel/AMD).

Supported: Intel Haswell (4th Gen) or newer, AMD Ryzen.

Not currently supported: ARM devices (Raspberry Pi, Apple Silicon, Android) require a port to NEON intrinsics.

Note: The architectural efficiency (250MB RAM usage) makes it theoretically ideal for Edge devices, but this specific demo code is optimized for x86 desktops.

## Usage Instructions

### 1. Exporting the Model

Use the `scripts/export_bin.py` script to convert your PyTorch/JAX checkpoints to the optimized C++ binary format.

**Arguments:**

- `--version`: Model version to export (`1b` or `250m`).
- `--ckpt_path`: Path to the checkpoint file (`.msgpack`).
- `--output_name`: Output binary filename.

#### Example for 1B version:

```bash
python3 scripts/export_bin.py --version 1b --ckpt_path ./bitmamba_1b.msgpack --output_name bitmamba_1b.bin
```

#### Example for 250M version:

```bash
python3 scripts/export_bin.py --version 250m --ckpt_path ./bitmamba_250m.msgpack --output_name bitmamba_250m.bin
```

---

### 2. Compile the C++ Inference Engine

### Option 1: Using CMake (Recommended)

Ensure you have CMake installed (sudo apt install cmake or equivalent).

```bash
cmake -B build
cmake --build build
```

The executable will be located at `build/bitmamba`.

### Option 2: Quick Build (Manual)

If you prefer `g++`:

```bash
g++ -O3 -march=native -fopenmp -Iinclude -Isrc -o bitmamba examples/main.cpp src/*.cpp
```

### 3. Running Inference

#### 3.1 Download Weights (from Hugging Face)

BitMamba-2 1B

```bash
wget https://huggingface.co/Zhayr1/BitMamba-2-1B/resolve/main/bitmamba_cpp/bitmamba_1b.bin
```

BitMamba-2 0.25B

```bash
wget https://huggingface.co/Zhayr1/BitMamba-2-0.25B/resolve/main/bitmamba_cpp/bitmamba_255m.bin
```

Once you have the binary model (`.bin`) and the compiled executable, use the exported binary to run inference.

**Example command:**

```bash
./build/bitmamba <model.bin> "<prompt_tokens>" <mode> <temp> <repeat_penalty> <top_p> <top_k> <max_tokens>
```

#### Practical Example:

##### CMake Build

Tokenizer mode:

```bash
./build/bitmamba bitmamba_1b.bin "Hello, I am" tokenizer 0.7 1.1 0.05 0.9 40 200
```

Raw mode:

```bash
./build/bitmamba bitmamba_1b.bin "15496 11 314 716" raw 0.7 1.1 0.05 0.9 40 200
```

##### Manual Build

Tokenizer mode:

```bash
./bitmamba bitmamba_1b.bin "Hello, I am" tokenizer 0.7 1.1 0.05 0.9 40 200
```

Raw mode:

```bash
./bitmamba bitmamba_1b.bin "15496 11 314 716" raw 0.7 1.1 0.05 0.9 40 200
```

⚠️ IMPORTANT: the tokenizer.bin file must be in the same directory as the bitmamba compiled executable.

_This command runs the `bitmamba_1b.bin` model with a tokenized prompt, temperature 0.7, repetition penalty 1.1, generating 200 tokens._

### 4. Decoding Tokens

If you use raw mode, you can use the `scripts/decoder.py` script to convert token IDs back into text.

**Usage:**

```bash
python scripts/decoder.py "tokens"
```

**Example:**

```bash
python scripts/decoder.py "15496 11 314 716"
```

### TODO

- Future Work: Add ARM/NEON support for Raspberry Pi deployment.

## Python Inference Evaluation test

Use the `scripts/fast_inference.py` script to evaluate the models:

### Get the weights

Weights for 250M version:

```bash
wget https://huggingface.co/Zhayr1/BitMamba-2-0.25B/resolve/main/jax_weights/bitmamba_255m.msgpack
```

Weights for 1B version:

```bash
wget https://huggingface.co/Zhayr1/BitMamba-2-1B/resolve/main/jax_weights/bit_mamba_1b.msgpack
```

### 250M Version

```bash
python scripts/fast_inference.py --ckpt bitmamba_255m.msgpack --version 250m --eval
```

### 1B Version

```bash
python scripts/fast_inference.py --ckpt bit_mamba_1b.msgpack --version 1b --eval
```
