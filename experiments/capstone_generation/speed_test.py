"""Speed comparison: dynamic halting vs fixed depth."""
import torch
import time
from core.recurrent_depth import RecurrentDepthConfig, RecurrentDepthLM
from core.generation import generate_with_halting

def main():
    print("Loading checkpoint...", flush=True)
    ck = torch.load(
        "experiments/capstone_generation/checkpoint.pt",
        map_location="cuda",
        weights_only=False,
    )
    cfg = ck["model_config"]
    config = RecurrentDepthConfig(
        context_size=cfg["context_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        ff_dim=cfg["ff_dim"],
        iterations=cfg["iterations"],
        temperature=cfg["temperature"],
        dropout=cfg.get("dropout", 0.0),
        normalize=cfg.get("normalize", True),
    )
    model = RecurrentDepthLM(vocab_size=int(ck["vocab_size"]), config=config).cuda()
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    stoi = ck["char_to_idx"]
    itos = {int(k): v for k, v in ck["idx_to_char"].items()}

    # Warmup
    print("Warmup...", flush=True)
    for _ in range(3):
        generate_with_halting(
            model, prompt="The ", stoi=stoi, itos=itos,
            length=50, threshold=0.02, temperature=0.7,
        )

    # Measure dynamic halting
    print("Measuring dynamic halting...", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        r = generate_with_halting(
            model, prompt="The ", stoi=stoi, itos=itos,
            length=200, threshold=0.02, temperature=0.7,
        )
    torch.cuda.synchronize()
    dyn = (time.perf_counter() - t0) / 5
    dd = sum(r.halting_depths) / len(r.halting_depths)

    # Measure fixed depth (threshold=0 means run all iterations)
    print("Measuring fixed depth...", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        rf = generate_with_halting(
            model, prompt="The ", stoi=stoi, itos=itos,
            length=200, threshold=0.0, temperature=0.7,
        )
    torch.cuda.synchronize()
    fix = (time.perf_counter() - t0) / 5
    fd = sum(rf.halting_depths) / len(rf.halting_depths)

    # Results
    results = [
        f"Dynamic (eps=0.02): {dyn:.3f}s, mean depth {dd:.2f}",
        f"Fixed (all {config.iterations}):     {fix:.3f}s, mean depth {fd:.2f}",
        f"Speedup: {fix/dyn:.2f}x",
        f"Compute savings: {1 - dd/fd:.1%}",
    ]
    for line in results:
        print(line, flush=True)

    with open("experiments/capstone_generation/speed_test.txt", "w") as f:
        f.write("\n".join(results) + "\n")


if __name__ == "__main__":
    main()
