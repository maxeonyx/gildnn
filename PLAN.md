# Plan

Immediate checklist. What's next, what I'll do based on each outcome. For the bigger picture, read VISION.md.

## Current state

**The spectator problem is robust and hard to break.** Five WikiText-103 experiments (all ctx=32, d=256, 20K steps) converge on the same conclusion: under the corrected architecture (only block 0 receives tokens), upper blocks cannot be made useful. Specifically:

| Intervention | Result | What it tested |
|---|---|---|
| Baseline (corrected vs old) | C≈A, B wins -0.101 | Multi-block only helps as ensemble |
| Aux prediction loss | NULL (+0.003) | Gradient signal isn't the bottleneck |
| Equal readout | HURTS (+0.024) | Readout collapse isn't the bottleneck |
| Temporal window (k=8) | NULL (+0.003) | Lower-neighbor trajectory isn't enough |

**Diagnosis:** The residual stream alone is insufficient for inter-block information propagation. Upper blocks CAN learn (aux losses decrease to 2.9) but what they learn isn't more useful for next-token prediction than what block 0 already provides. Temporal history helps block 1 slightly (readout 14%→18%) but doesn't cascade up.

**Emerging hypothesis:** At ctx=32, the prediction task may be too simple for hierarchical processing. A single feedforward with direct token access captures enough pattern. Hierarchy might only matter at longer contexts where slower blocks could model dependencies beyond a single block's horizon.

## Active

Nothing running. GPU free.

## Next — two candidate directions

### Option 1: Scale context to test the hierarchy-needs-longer-context hypothesis
- Increase context from 32 to 128 or 256
- At longer contexts, slower blocks (rate 8 = processing every 8 steps) should genuinely capture patterns the fast block can't
- This tests whether the architecture becomes useful when the task actually requires multi-timescale processing
- Risk: longer context changes many things at once; may not isolate the effect

### Option 2: All-blocks-get-tokens but with local learning (true parallelism test)
- Use variant B architecture (every block sees tokens) — this is the only one that works
- Add stop-grad: each block trains on its OWN loss only, no global backprop
- Test whether blocks can learn INDEPENDENTLY and still combine usefully
- This is the closest thing to Max's actual research question about parallelism
- The question becomes: can an ensemble of independently-trained modules outperform a single module?

### Option 3: Transformer baseline (methodological debt)
- Build a transformer at matched compute (~3.6M params) on WikiText-103 ctx=32
- Tell us where 1.83 sits relative to standard architectures
- Important context for interpreting all results

## Queue (lower priority)

- **Many more blocks (16/32)** — unlikely to help based on current evidence
- **Graph architecture** — Per [dictation 2026-05-23-5](dictations/2026-05-23-5.md)
- **Interface predictive-coding loss** — requires working architecture first
- Named/typed tensor dimensions
- Loop management tooling

## Key completed findings

All on WikiText-103 (ctx=32, d=256, 20K steps) unless noted.

| Experiment | Result | Notes |
|---|---|---|
| WikiText-103 baseline | B wins (-0.101), C≈A, D tiny edge | Spectator persists with more data |
| Local aux loss | **NULL** (+0.003) | Gradient signal isn't the problem |
| Equal readout | **HURTS** (+0.024) | Confirms information poverty |
| Temporal window (k=8) | **NULL** (+0.003) | History helps block 1 readout but no val_loss improvement |
| Bidirectional top-down | **HURTS** (+0.06-0.08) | TinyShakespeare |
| Architecture correction | **HURTS** (+0.034 vs single) | Upper blocks are spectators |
| Width scaling (d=256, 4-block, old arch) | **WINS** (-0.023) | Best TinyShakespeare result |
| Matched-FLOP multi-rate (old arch) | **WINS** (+0.019 avg) | 2/3 seeds better |
| CUDA Graph training | 10.86× speedup | GraphTrainer in core/ |
| Self-prediction | NULL | Zero task effect |
| Cosine LR | NULL | Dataset bottleneck |
