# BitMamba.cpp Library

This library is designed for efficient inference using **BitMamba2** models with **255M** and **1B** parameters. It implements support for quantization and BitNet-optimized architectures.

## Requirements and compatibility

⚠️ Hardware Requirements: This C++ implementation utilizes AVX2 SIMD instructions for high-performance inference on x86 CPUs (Intel/AMD).

Supported: Intel Haswell (4th Gen) or newer, AMD Ryzen.

Not currently supported: ARM devices (Raspberry Pi, Apple Silicon, Android) require a port to NEON intrinsics.

Note: The architectural efficiency (250MB RAM usage) makes it theoretically ideal for Edge devices, but this specific demo code is optimized for x86 desktops.

## Usage Instructions

### 1. Exporting the Model

Use the `export_bin.py` script to convert your PyTorch/JAX checkpoints to the optimized C++ binary format.

**Arguments:**

- `--version`: Model version to export (`1b` or `250m`).
- `--ckpt_path`: Path to the checkpoint file (`.msgpack`).
- `--output_name`: Output binary filename.

#### Example for 1B version:

```bash
python3 export_bin.py --version 1b --ckpt_path ./bitmamba_1b.msgpack --output_name bitmamba_1b.bin
```

#### Example for 250M version:

```bash
python3 export_bin.py --version 250m --ckpt_path ./bitmamba_250m.msgpack --output_name bitmamba_250m.bin
```

---

### 2. Compile the C++ Inference Engine

### Option 1: Quick Build (Linux/WSL)

If you have `g++` installed, you can compile directly:

```bash
g++ -O3 -march=native -fopenmp -o bitmamba bitmamba.cpp
```

### Option 2: Using CMake (Recommended for cross-platform)

Ensure you have CMake installed (sudo apt install cmake or equivalent).

```bash
mkdir build
cd build
cmake ..
make -j
```

### 3. Running Inference

Once you have the binary model (`.bin`) and the compiled executable, use the exported binary to run inference.

**Example command:**

```bash
./bitmamba <model.bin> "<prompt_tokens>" <temp> <repeat_penalty> <top_p> <top_k> <max_tokens> <seed>
```

#### Practical Example:

```bash
./bitmamba bitmamba_1b.bin "15496 11 314 716" 0.7 1.1 0.05 0.9 40 200
```

Or in build folder:

```bash
./bitmamba ../bitmamba_1b.bin "15496 11 314 716" 0.7 1.1 0.05 0.9 40 200
```

_This command runs the `bitmamba_1b.bin` model with a tokenized prompt, temperature 0.7, repetition penalty 1.1, generating 200 tokens._

### 4. Decoding Tokens

Use the `decoder.py` script to convert token IDs back into text.

**Usage:**

```bash
python decoder.py "tokens"
```

**Example:**

```bash
python decoder.py "15496 11 314 716"
```

### TODO

- Future Work: Add ARM/NEON support for Raspberry Pi deployment.

## Python Inference Evaluation test

Use the `fast_inference.py` script to evaluate the models:

### 250M Version

```bash
python fast_inference.py --ckpt bitmamba_250m.msgpack --version 250m --eval
```

### 1B Version

```bash
python fast_inference.py --ckpt bitmamba_1b.msgpack --version 1b --eval
```
