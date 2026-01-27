import torch
import torch.nn.functional as F
import numpy as np
from flax.serialization import msgpack_restore 
import re
import argparse
import types
from transformers import AutoTokenizer, TextStreamer # <--- Agrega TextStreamer
from bit_mamba_torch import BitMambaLM, BitLinear


# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_1B = {
    "vocab_size": 50257,
    "d_model": 2048,
    "n_layers": 32,
    "n_heads": 32,
}

CONFIG_255M = {
     "vocab_size": 50257,
     "d_model": 1024,
     "n_heads": 16,
     "n_layers": 24
}

CONFIG = CONFIG_255M

# --- LOADING FUNCTIONS ---
def jax_to_torch(param_name, jax_array):
    np_arr = np.array(jax_array, dtype=np.float32)
    tensor = torch.from_numpy(np_arr)
    if "kernel" in param_name and tensor.dim() == 2:
        return tensor.t()
    if "conv1d" in param_name and "kernel" in param_name and tensor.dim() == 3:
        return tensor.permute(2, 1, 0)
    return tensor

def load_checkpoint(model, path):
    print(f"📂 Reading JAX msgpack: {path} ...")
    with open(path, "rb") as f:
        state_dict = msgpack_restore(f.read())
    if 'params' in state_dict: state_dict = state_dict['params']
    
    print("🔄 Converting weights...")
    torch_state_dict = {}
    layer_pattern = re.compile(r".*BitMambaBlock_(\d+)$")

    for key, val in state_dict.items():
        if 'Embed_0' in key:
            torch_state_dict["embed.weight"] = jax_to_torch("embed", val['embedding'])
        elif 'RMSNorm_0' in key:
            torch_state_dict["norm_f.weight"] = jax_to_torch("norm", val['scale'])
        elif key == 'BitLinear_0':
            torch_state_dict["lm_head.weight"] = jax_to_torch("kernel", val['kernel'])
            if 'RMSNorm_0' in val:
                torch_state_dict["lm_head.norm.weight"] = jax_to_torch("norm", val['RMSNorm_0']['scale'])
        
        match = layer_pattern.match(key)
        if match:
            idx = match.group(1)
            base = f"layers.{idx}."
            if 'BitLinear_0' in val: 
                torch_state_dict[base + "in_proj.weight"] = jax_to_torch("kernel", val['BitLinear_0']['kernel'])
                if 'RMSNorm_0' in val['BitLinear_0']:
                    torch_state_dict[base + "in_proj.norm.weight"] = jax_to_torch("norm", val['BitLinear_0']['RMSNorm_0']['scale'])
            if 'Conv_0' in val:
                if 'bias' in val['Conv_0']:
                    torch_state_dict[base + "conv1d.bias"] = jax_to_torch("bias", val['Conv_0']['bias'])
                torch_state_dict[base + "conv1d.weight"] = jax_to_torch("conv1d.kernel", val['Conv_0']['kernel'])
            if 'BitLinear_1' in val: 
                torch_state_dict[base + "out_proj.weight"] = jax_to_torch("kernel", val['BitLinear_1']['kernel'])
                if 'RMSNorm_0' in val['BitLinear_1']:
                    torch_state_dict[base + "out_proj.norm.weight"] = jax_to_torch("norm", val['BitLinear_1']['RMSNorm_0']['scale'])
            if 'dt_bias' in val: torch_state_dict[base + "dt_bias"] = jax_to_torch("p", val['dt_bias'])
            if 'A_log' in val: torch_state_dict[base + "A_log"] = jax_to_torch("p", val['A_log'])
            if 'D' in val: torch_state_dict[base + "D"] = jax_to_torch("p", val['D'])

    model.load_state_dict(torch_state_dict, strict=False)
    print("✅ Load completed.")

# --- OPTIMIZATION: WEIGHT FUSION ---
def bitlinear_fused_forward(self, x):
    if hasattr(self, 'norm') and self.norm is not None:
        x = self.norm(x)
    scale_x = 127.0 / x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
    x_quant = (x * scale_x).round().clamp(-128, 127) / scale_x
    return F.linear(x_quant, self.weight, self.bias)

def fuse_bitnet_weights(model, BitLinear_cls):
    print("🧊 Freezing (Fusing) BitNet weights for fast inference...")
    fused_count = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, BitLinear_cls) or module.__class__.__name__ == 'BitLinear':
                w = module.weight
                scale_w = 1.0 / w.abs().mean().clamp(min=1e-5)
                w_quant = (w * scale_w).round().clamp(-1, 1) / scale_w
                module.weight.data = w_quant
                module.forward = types.MethodType(bitlinear_fused_forward, module)
                fused_count += 1
    print(f"✅ Optimized {fused_count} BitLinear layers.")

# --- QUALITY LOGIC ---
def apply_repetition_penalty(logits, sequences, penalty):
    for i in range(logits.shape[0]):
        previous_tokens = sequences[i].unique()
        score = torch.gather(logits[i, :], 0, previous_tokens)
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits[i, :].scatter_(0, previous_tokens, score)
    return logits

# --- GENERATE FUNCTION (CORRECTED) ---
@torch.inference_mode()
def generate_topk(model, tokenizer, prompt, max_new_tokens=100, temperature=0.6, top_k=40, repetition_penalty=1.2):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    caches = model.init_caches(input_ids.shape[0], DEVICE)
    curr_token = input_ids[:, -1]
    
    print(f"🤖 BitMamba: {prompt}", end="", flush=True)
    
    # BEGIN CHANGE: Initialize Streamer
    # skip_prompt=True so it doesn't repeat what you already printed above
    streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)
    # END CHANGE

    # Prefill
    for i in range(input_ids.shape[1] - 1):
        _, caches = model.step(input_ids[:, i], caches)
        
    # Generation
    for _ in range(max_new_tokens):
        logits, caches = model.step(curr_token, caches)
        
        # 1. Repetition Penalty
        if repetition_penalty > 1.0:
            logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)
        
        # 2. Temperature
        logits = logits / temperature
        
        # 3. Top-K
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # 4. Sampling
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(0)
        
        # --- CRITICAL CHANGE HERE ---
        # Instead of manual decode + print, we give the token to the streamer
        # The streamer knows to wait if the byte is incomplete.
        streamer.put(next_token.unsqueeze(0)) 
        # ---------------------------
        
        input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
        
        if next_token.item() == tokenizer.eos_token_id: break
        curr_token = next_token
    
    # Clean the final buffer (in case something is pending)
    streamer.end()
    print("\n" + "-"*40)

#--- GENERATE FUNCTION (Reusable) ---
@torch.inference_mode()
def generate_topk_v1(model, tokenizer, prompt, max_new_tokens=100, temperature=0.6, top_k=40, repetition_penalty=1.2):
    """
    Single generation function (non-interactive) optimized.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    caches = model.init_caches(input_ids.shape[0], DEVICE)
    curr_token = input_ids[:, -1]
    
    print(f"🤖 BitMamba: {prompt}", end="", flush=True)
    
    # Prefill
    for i in range(input_ids.shape[1] - 1):
        _, caches = model.step(input_ids[:, i], caches)
        
    # Generation
    for _ in range(max_new_tokens):
        logits, caches = model.step(curr_token, caches)
        
        # 1. Repetition Penalty
        if repetition_penalty > 1.0:
            logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)
        
        # 2. Temperature
        logits = logits / temperature
        
        # 3. Top-K
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # 4. Sampling
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(0)
        
        word = tokenizer.decode([next_token.item()])
        print(word, end="", flush=True)
        
        # Concatenate for penalty history
        input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
        
        if next_token.item() == tokenizer.eos_token_id: break
        curr_token = next_token
    print("\n" + "-"*40)

# --- EXECUTION MODES ---
@torch.inference_mode()
def chat_stream(model, tokenizer, temperature=0.6, top_k=40, repetition_penalty=1.2):
    print("\n💬 Interactive Chat Mode")
    while True:
        prompt = input("\n👤 User: ")
        if prompt.lower() in ["exit", "quit"]: break
        # Reuse logic, although here we duplicate slightly to keep the interactive loop fluid
        generate_topk(model, tokenizer, prompt, 200, temperature, top_k, repetition_penalty)

@torch.inference_mode()
def benchmark_speed(model, tokenizer, prompt="The future of AI is", steps=200):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    caches = model.init_caches(input_ids.shape[0], DEVICE)
    print(f"\n🏎️  Starting FUSED Benchmark ({steps} tokens)...")
    
    for i in range(input_ids.shape[1] - 1):
        _, caches = model.step(input_ids[:, i], caches)
    curr_token = input_ids[:, -1]
    
    print("🔥 Warming up caches...")
    for _ in range(10):
        model.step(curr_token, caches)

    if DEVICE == "cuda": torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(steps):
        logits, caches = model.step(curr_token, caches)
        curr_token = torch.argmax(logits, dim=-1) # Greedy for pure speed
    
    end_event.record()
    if DEVICE == "cuda": torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    tokens_per_sec = steps / (elapsed_time_ms / 1000)
    print(f"⚡ Real Speed: {tokens_per_sec:.2f} tokens/sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint file (.msgpack)")
    parser.add_argument("--version", type=str, choices=["1b", "250m"], required=True, help="Model version: '1b' or '250m'")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode")
    parser.add_argument("--eval", action="store_true", help="Run evaluation test suite")
    parser.add_argument("--temp", type=float, default=0.6, help="Temperature")
    parser.add_argument("--topk", type=int, default=40, help="Top-K")
    parser.add_argument("--penalty", type=float, default=1.2, help="Repetition Penalty")
    parser.add_argument("--v1", action="store_true", help="Use generator version 1")
    
    args = parser.parse_args()

    # Dynamic Import based on version
    if args.version == "1b":
        config = CONFIG_1B
    else:
        config = CONFIG_255M

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"🏗️ Creating model ({args.version})...")
    model = BitMambaLM(**config)
    load_checkpoint(model, args.ckpt_path)

    print(f"🚀 Moving to {DEVICE} (float16)...")
    model.to(DEVICE).to(torch.float16)
    
    # APPLY SPEED OPTIMIZATION
    fuse_bitnet_weights(model, BitLinear)
    model.eval()

    if args.chat:
        chat_stream(model, tokenizer, args.temp, args.topk, args.penalty)
    
    elif args.eval:
        print("\n📊 RUNNING EVALUATION SUITE")
        # Configuration optimized for coherence
        kwargs = {
            "temperature": 0.6, 
            "repetition_penalty": 1.2, 
            "top_k": 40,
            "max_new_tokens": 100
        }

        if args.v1:
            generate_topk_v1(model, tokenizer, "Artificial intelligence is", **kwargs)
            generate_topk_v1(model, tokenizer, "Water boils at", **kwargs)
            generate_topk_v1(model, tokenizer, "The capital of France is", **kwargs)
            generate_topk_v1(model, tokenizer, "The Eiffel Tower is in", **kwargs)
            generate_topk_v1(model, tokenizer, "The largest planet in the solar system is", **kwargs)
            generate_topk_v1(model, tokenizer, "The tallest mammal is", **kwargs)
            generate_topk_v1(model, tokenizer, "The world's highest mountain is", **kwargs)
            generate_topk_v1(model, tokenizer, "The world's deepest ocean is", **kwargs)
            generate_topk_v1(model, tokenizer, "def add_two_numbers(a, b):", **kwargs)
        else:
            generate_topk(model, tokenizer, "Artificial intelligence is", **kwargs)
            generate_topk(model, tokenizer, "Water boils at", **kwargs)
            generate_topk(model, tokenizer, "The capital of France is", **kwargs)
            generate_topk(model, tokenizer, "The Eiffel Tower is in", **kwargs)
            generate_topk(model, tokenizer, "The largest planet in the solar system is", **kwargs)
            generate_topk(model, tokenizer, "The tallest mammal is", **kwargs)
            generate_topk(model, tokenizer, "The world's highest mountain is", **kwargs)
            generate_topk(model, tokenizer, "The world's deepest ocean is", **kwargs)
            generate_topk(model, tokenizer, "def add_two_numbers(a, b):", **kwargs)
        
    else:
        benchmark_speed(model, tokenizer)
