"""Temporal-window experiment analysis.

Run this when all 9 runs (3 variants x 3 seeds) are complete.
Implements the pre-registered interpretation from research/questions/temporal-window/README.md.
"""
import json
from pathlib import Path
from collections import defaultdict

LOG_PATH = Path(r"C:\Users\maxeo\gildnn\experiments\temporal-window\artifacts\run.jsonl")

def load_finals():
    entries = [json.loads(line) for line in LOG_PATH.read_text().splitlines() if line.strip()]
    finals = [e for e in entries if e.get("step") == 20000]
    return finals

def analyze():
    finals = load_finals()
    
    by_variant = defaultdict(list)
    for f in finals:
        by_variant[f["variant"]].append(f)
    
    print(f"=== Temporal Window Results ===")
    print(f"Completed runs: {len(finals)} / 9 expected")
    print()
    
    for v in ["B0", "H8", "C8"]:
        seeds = sorted([f["seed"] for f in by_variant[v]])
        print(f"  {v}: seeds {seeds} ({len(seeds)}/3)")
    
    if len(finals) < 9:
        print(f"\nNOT COMPLETE - only {len(finals)}/9 runs finished. Partial results below.\n")
    
    print()
    print("=== Per-seed val_loss ===")
    print(f"{'Seed':<6} {'B0':<10} {'H8':<10} {'C8':<10} {'d_traj':<10} {'d_aug':<10} {'d_ctrl':<10}")
    print("-" * 66)
    
    seeds = sorted(set(f["seed"] for f in finals))
    deltas_traj = []
    deltas_aug = []
    deltas_ctrl = []
    
    for seed in seeds:
        b0 = next((f["val_loss"] for f in finals if f["variant"] == "B0" and f["seed"] == seed), None)
        h8 = next((f["val_loss"] for f in finals if f["variant"] == "H8" and f["seed"] == seed), None)
        c8 = next((f["val_loss"] for f in finals if f["variant"] == "C8" and f["seed"] == seed), None)
        
        dt = (c8 - h8) if (c8 is not None and h8 is not None) else None
        da = (b0 - h8) if (b0 is not None and h8 is not None) else None
        dc = (b0 - c8) if (b0 is not None and c8 is not None) else None
        
        if dt is not None: deltas_traj.append(dt)
        if da is not None: deltas_aug.append(da)
        if dc is not None: deltas_ctrl.append(dc)
        
        b0_s = f"{b0:.4f}" if b0 else "---"
        h8_s = f"{h8:.4f}" if h8 else "---"
        c8_s = f"{c8:.4f}" if c8 else "---"
        dt_s = f"{dt:+.4f}" if dt else "---"
        da_s = f"{da:+.4f}" if da else "---"
        dc_s = f"{dc:+.4f}" if dc else "---"
        
        print(f"{seed:<6} {b0_s:<10} {h8_s:<10} {c8_s:<10} {dt_s:<10} {da_s:<10} {dc_s:<10}")
    
    print()
    
    if deltas_traj:
        mean_dt = sum(deltas_traj) / len(deltas_traj)
        mean_da = sum(deltas_aug) / len(deltas_aug)
        mean_dc = sum(deltas_ctrl) / len(deltas_ctrl)
        print(f"Mean d_trajectory (C8-H8): {mean_dt:+.4f}  (threshold: >=+0.015 for clear)")
        print(f"Mean d_augmented  (B0-H8): {mean_da:+.4f}  (threshold: >=+0.015)")
        print(f"Mean d_control    (B0-C8): {mean_dc:+.4f}  (threshold: >=+0.015)")
        print()
        
        all_traj_positive = all(d > 0 for d in deltas_traj)
        all_aug_positive = all(d > 0 for d in deltas_aug)
        
        print(f"Concordance (all seeds same sign):")
        print(f"  d_trajectory: {'YES all positive' if all_traj_positive else 'NO - not all positive'}")
        print(f"  d_augmented:  {'YES all positive' if all_aug_positive else 'NO - not all positive'}")
        print()
        
        print("=== Pre-registered Interpretation ===")
        if mean_dt >= 0.015 and all_traj_positive:
            print("TRAJECTORY CLEARLY HELPS")
            print("   H8 beats C8 by >=0.015, all seeds concordant.")
            print("   Next: 4-block scaling with window + readout_mode=all")
        elif abs(mean_dt) < 0.005 and mean_da >= 0.015 and mean_dc >= 0.015:
            print("AUGMENTATION HELPS BUT TRAJECTORY DOES NOT MATTER")
            print("   H8 approx C8 but both beat B0. It is capacity, not history.")
            print("   Next: simpler capacity/interface fixes")
        elif abs(mean_da) < 0.005 and abs(mean_dc) < 0.005:
            print("NOTHING HELPS")
            print("   Neither history nor control augmentation improves over B0.")
            print("   Pivot to surrogate architecture work")
        elif mean_da <= -0.005:
            print("NEGATIVE - auxiliary branch hurts")
            print("   Investigate training dynamics")
        else:
            print("AMBIGUOUS - does not clearly fit any pre-registered category")
            print(f"   d_trajectory={mean_dt:+.4f}, d_augmented={mean_da:+.4f}, d_control={mean_dc:+.4f}")
            if not all_traj_positive and mean_dt >= 0.015:
                print("   Mean is large but seeds are discordant - need more seeds")
    
    print()
    for v in ["B0", "H8", "C8"]:
        losses = [f["val_loss"] for f in by_variant[v]]
        if losses:
            mean = sum(losses) / len(losses)
            std = (sum((x - mean)**2 for x in losses) / max(len(losses)-1, 1))**0.5
            print(f"{v}: {mean:.4f} +/- {std:.4f} ({len(losses)} seeds)")

if __name__ == "__main__":
    analyze()
