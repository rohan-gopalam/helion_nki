import torch;
import torch_neuronx;
import torch_xla.core.xla_model as xm;
import helion;
import helion.language as hl;

custom_config = helion.Config(block_sizes=[128, 128, 128], platform_target="inf1");

@helion.kernel(
    backend="nki", 
    config=custom_config,
    autotune_effort="none", 
)
def simple_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size();
    k2, n = b.size();
    out = torch.empty((m, n), dtype=a.dtype, device=a.device);
    
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros((tile_m, tile_n), dtype=torch.float32);
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n]);
        out[tile_m, tile_n] = acc;
        
    return out;

print("Acquire Trainium device...");
device = xm.xla_device();

print("Creating tensors on trainium...");
a = torch.randn(128, 128, device=device);
b = torch.randn(128, 128, device=device);

print("Attempt to execute kernel...");
result = simple_matmul(a, b);

# force XLA to finish
xm.mark_step();

print(f"Kernel finished. Output shape: {result.shape}");

