import torch
import re
import numpy as np
from flax.serialization import msgpack_restore 

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