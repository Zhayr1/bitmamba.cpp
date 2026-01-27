import torch
import torch.nn as nn
import torch.nn.functional as F

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.norm = nn.RMSNorm(in_features, eps=1e-6)

    def forward(self, x):
        x = x.to(self.weight.dtype)
        x_norm = self.norm(x)

        scale_x = 127.0 / x_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_quant = (x_norm * scale_x).round().clamp(-128, 127) / scale_x
        
        scale_w = 1.0 / self.weight.abs().mean().clamp(min=1e-5)
        w_quant = (self.weight * scale_w).round().clamp(-1.0, 1.0) / scale_w
        
        return F.linear(x_quant, w_quant, self.bias)

class BitMambaBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_inner = int(expand * d_model)
        self.head_dim = self.d_inner // n_heads
        self.d_conv = d_conv
        
        dim_proj = 2 * self.d_inner + 3 * self.n_heads
        self.in_proj = BitLinear(d_model, dim_proj, bias=False)
        
        # Conv1d with padding=0 for manual inference
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, 
            out_channels=self.d_inner, 
            kernel_size=d_conv, 
            groups=self.d_inner, 
            padding=0, 
            bias=True
        )
        
        self.out_proj = BitLinear(self.d_inner, d_model, bias=False)
        
        self.dt_bias = nn.Parameter(torch.zeros(n_heads))
        self.A_log = nn.Parameter(torch.zeros(n_heads))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward_step(self, u, cache):
        # 1. Projection
        zxbcdt = self.in_proj(u)
        
        sizes = [self.d_inner, self.d_inner, self.n_heads, self.n_heads, self.n_heads]
        z, x_in, B_val, C_val, dt = torch.split(zxbcdt, sizes, dim=-1)
        
        # 2. Manual Convolution
        conv_state = cache['conv']
        x_expanded = x_in.unsqueeze(-1)
        
        new_conv_state = torch.cat([conv_state[:, :, 1:], x_expanded], dim=-1)
        conv_window = torch.cat([conv_state, x_expanded], dim=-1)
        
        # Convolution without padding (exact)
        x_conv_out = F.conv1d(
            conv_window, 
            self.conv1d.weight, 
            self.conv1d.bias, 
            padding=0, 
            groups=self.d_inner
        )
        
        x_t = F.silu(x_conv_out.squeeze(-1))

        # 3. SSM
        x_reshaped = x_t.view(-1, self.n_heads, self.head_dim)
        
        dt = F.softplus(dt + self.dt_bias).unsqueeze(-1)
        A = -torch.exp(self.A_log).view(1, self.n_heads, 1)
        decay = torch.exp(A * dt)
        u_ssm = x_reshaped * B_val.unsqueeze(-1) * dt
        
        h_new = cache['ssm'] * decay + u_ssm
        
        y = h_new * C_val.unsqueeze(-1)
        y = y.view(-1, self.d_inner)
        
        y = y + x_t * self.D
        y = y * F.silu(z)
        
        out = self.out_proj(y)
        
        return out, {'conv': new_conv_state, 'ssm': h_new}
        
    def init_cache(self, batch_size, device):
        # Automatically detect dtype
        dtype = self.in_proj.weight.dtype 
        return {
            'conv': torch.zeros(batch_size, self.d_inner, self.d_conv - 1, device=device, dtype=dtype),
            'ssm': torch.zeros(batch_size, self.n_heads, self.head_dim, device=device, dtype=dtype)
        }

class BitMambaLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            BitMambaBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm_f = nn.RMSNorm(d_model, eps=1e-6)
        self.lm_head = BitLinear(d_model, vocab_size, bias=False)

    def step(self, input_ids, caches):
        x = self.embed(input_ids)
        new_caches = []
        
        for i, layer in enumerate(self.layers):
            # Save input as residual
            residual = x 
            
            # Execute layer
            layer_out, new_c = layer.forward_step(x, caches[i])
            
            # --- CRITICAL CORRECTION: ADD RESIDUAL ---
            x = residual + layer_out 
            
            new_caches.append(new_c)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits, new_caches

    def init_caches(self, batch_size, device):
        return [layer.init_cache(batch_size, device) for layer in self.layers]