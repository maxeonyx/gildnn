"""Quick overfit sanity check for partial detach model."""
import torch
from torch.nn import functional as F
from core.fixed_window_char import set_seed
from experiments.residual_stream_time_partial_detach.model import (
    PartialDetachCharModel, PartialDetachConfig,
)

set_seed(42)
device = torch.device("cuda")
config = PartialDetachConfig(
    context_size=32, d_model=64, feedforward_dim=256,
    detach_every_n=4, temporal_window=4, num_heads=4,
)
model = PartialDetachCharModel(vocab_size=65, config=config).to(device)
inputs = torch.randint(0, 65, (16, 32), device=device)
targets = torch.randint(0, 65, (16,), device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

for step in range(200):
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"step {step}: loss={loss.item():.4f}", flush=True)

acc = (model(inputs).argmax(dim=1) == targets).float().mean().item()
print(f"Final: acc={acc:.3f} loss={loss.item():.6f}", flush=True)
print("OVERFIT_PASS" if acc > 0.9 else "OVERFIT_FAIL", flush=True)
