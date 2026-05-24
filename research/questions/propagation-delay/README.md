# Propagation-Delay 2-Block: Does the Interior Block Learn?

**Pathway:** 3 (Local Learning), also connects to 1 (Wide Recurrent)
**Dictation grounding:** [2026-05-22-7](../../../dictations/2026-05-22-7.md) (async blocks, stale communication), [2026-05-24-4](../../../dictations/2026-05-24-4.md) (module independence)
**Status:** Conceptual clarification (pre-implementation)

---

## The question

In a 2-block architecture with propagation delay (each block sees the other's PREVIOUS-timestep state), can an interior block — one without direct access to the task loss — learn useful representations using only a local signal?

This is the **first correct test** of local learning. All prior experiments (strict-local, neighborhood-local, variants A-J) were tested WITHOUT propagation delay, making them architecturally incoherent relative to the project's actual thesis.

---

## Architecture

Two blocks communicating bidirectionally with 1-timestep delay. Uses `ParallelDiagonalModel` from `core/model.py` with `topology="top_down_to_first"`, `internal_steps=1`.

**How the code works (verified against `core/model.py`):**

The model processes all `context_size` tokens sequentially, building up per-block state. At the end of the sequence, a single readout from block 0's final state produces logits for next-token prediction.

```
previous_states = [zeros, zeros]  # [block_A, block_B]

for each token t in sequence:
    # Inject token into block A only
    seeded_A = mix_token(previous_A, token_embedding[t])
    seeded_B = previous_B  # no token injection

    # Block A: gets lateral from block B's PREVIOUS state
    lateral_A = maybe_detach(previous_B)
    input_A = 0.5 * (seeded_A + lateral_A)
    state_A = mix_block(input_A, ffn_A(input_A))

    # Block B: gets lateral from block A's PREVIOUS state
    lateral_B = maybe_detach(previous_A)
    input_B = 0.5 * (seeded_B + lateral_B)
    state_B = mix_block(input_B, ffn_B(input_B))

    previous_states = [state_A, state_B]

# Single final readout (NOT per-token)
logits = output_head(state_A)           # readout_mode="first"
task_loss = cross_entropy(logits, next_token_target)
```

Key properties:
- Block A = **edge block**: gets tokens, produces final task output, always has full gradient
- Block B = **interior block**: no direct token access, no direct task output
- Communication is bidirectional but always 1-timestep stale (guaranteed by `internal_steps=1`)
- Block B's contribution to Block A arrives at the NEXT timestep (via lateral delay)
- The task loss is attached only at the final position — this is next-character prediction from a fixed context window

**Why `internal_steps=1` matters:** With `internal_steps > 1`, later internal steps use `current_states` (not `previous_states`), eliminating the propagation delay within a timestep. Fixing `internal_steps=1` ensures the delay is always exactly 1 token-step.

---

## Hypotheses

**H1 (ceiling):** With full backprop through lateral connections, Block B becomes load-bearing (ablating it degrades loss by ≥0.02 nats).

**H2 (floor):** With stop-gradient on laterals + a local CE head on Block B, Block B still becomes load-bearing (ablation degrades by ≥0.02 nats).

**H3 (spectator):** Even with full backprop, Block B does not become load-bearing (ablation effect <0.02 nats).

**Load-bearing threshold:** ≥0.02 nats = load-bearing. <0.02 nats = spectator. There is no gray zone — 0.02 is the bright line. Rationale: the stale-read quality cost was +0.005 ± 0.005 nats (noise floor). A 0.02 effect is 4× the noise floor and thus unambiguously above chance.

---

## Interpretation of H3 (if spectator)

If Block B is a spectator even with full backprop, this does NOT uniquely identify an architecture failure. Possible explanations include:

- Task too easy for 1 block at this scale (block A already sufficient)
- Optimization budget insufficient for B to develop useful features
- Final-step-only supervision provides too weak a gradient signal for B
- d_model=72 / context=32 not sensitive enough to benefit from a second block

H3 would mean: "at THIS scale and task, the architecture doesn't use both blocks." It would NOT prove Pathway 3 is dead — it would indicate we need to test at a larger scale where 1 block is insufficient, OR that the topology needs revision. Potential redirects: try longer context (where 1 block is provably insufficient), or try Pathway 10 (topology/broadcast).

---

## Experimental conditions

| # | Condition | `num_blocks` | `detach_lateral` | Local head on B | Purpose |
|---|---|---|---|---|---|
| 1 | Single-block baseline | 1 | n/a (topology="upward") | n/a | What 1 block achieves alone |
| 2 | 2-block, full backprop | 2 | False | Yes (unused, zero weight in loss) | Ceiling: can B help with global gradient? |
| 3 | 2-block, stop-gradient + local CE | 2 | True | Yes (active, weighted in loss) | Floor: can B help with local signal only? |

**Parameter matching:** Conditions 2 and 3 have identical parameters — both carry the local head on B, but condition 2 gives it zero weight in the loss (it exists only so parameter counts match). Condition 1 has fewer parameters (1 FFN + 1 head vs 2 FFN + 2 heads). This is intentional — we're asking "does adding a second block help?" not "is 2 blocks better than 1 block at matched params."

---

## Local CE head specification

- **Architecture:** `nn.Linear(d_model, vocab_size)` — a separate LM head applied to Block B's state
- **Input:** Block B's final state after processing all `context_size` tokens (NOT per-token; mirrors the global readout structure)
- **Target:** Same as the global task — predict the next token
- **Total loss (condition 3):** `task_CE(block_A) + λ * local_CE(block_B)` with `λ = 1.0` (equal weighting). Rationale: both heads predict the same target; no reason to down-weight one.
- **Total loss (condition 2):** `task_CE(block_A)` only. The local head exists but gets zero loss weight. Its parameters update only through weight decay.

**Gradient flow under condition 3 (stop-gradient):**
- Task CE gradient → block A parameters, output head. Cannot reach B (lateral detached).
- Local CE gradient → block B parameters, local head. Cannot reach A (lateral detached).
- Block B still has temporal recurrence across timesteps (its own `previous_states[1]` feeds into the next timestep). The local loss gradient flows through B's full temporal trajectory — this is NOT single-step local. It's "local" in the sense of not crossing the block boundary, but it does backprop through B's own time steps.

---

## Load-bearing ablation methodology

**Method: batch-shuffle ablation.** At eval time, randomly permute Block B's lateral output across the batch dimension before it enters Block A.

Why batch-shuffle instead of zeroing:
- The code computes `input_A = 0.5 * (seeded_A + lateral_A)`. Zeroing `lateral_A` halves input magnitude, confounding "B carried information" with "A is sensitive to magnitude."
- Batch-shuffling preserves the scale and distribution of B's output but destroys token-specific information.
- If loss increases under shuffle: B was carrying useful token-specific state to A.

**Implementation:** After training, run eval twice:
1. Normal forward pass → baseline val_loss
2. Forward pass with `previous_states[1]` shuffled across batch dim at each timestep → ablated val_loss
3. Load-bearing effect = ablated - baseline

---

## Model configuration

Using `ParallelDiagonalModel` from `core/model.py`:
- `topology="top_down_to_first"` (bidirectional lateral communication)
- `readout_mode="first"` (Block 0/A produces output)
- `token_injection="block0"` (only edge block gets tokens)
- `internal_steps=1` (ensures 1-timestep delay is exact)
- `d_model=72`, `feedforward_dim=288` (matching prior tiny-rung experiments)
- `context_size=32` (TinyShakespeare, matching prior baselines)
- 3 seeds per condition (seeds 42, 137, 2024)
- Training: 3000 steps, batch_size=64, AdamW lr=3e-3, weight_decay=0.01

---

## Results (2026-05-25)

### Val loss comparison

| Condition | Seed 42 | Seed 137 | Seed 2024 | Mean | Params |
|---|---|---|---|---|---|
| Single-block | 1.875 | 1.873 | 1.899 | **1.882** | 53K |
| 2-block gated (zero-init) | 1.912 | 1.876 | 1.887 | **1.892** | 99K |
| 2-block hardcoded 0.5 | 2.008 | 1.963 | 2.048 | **2.006** | 99K |
| 2-block stop-gradient | 2.820 | 4.889 | 12.756 | **6.822** | 99K |

### Batch-shuffle ablation (block B load-bearing test)

| Condition | Seed 42 | Seed 137 | Seed 2024 | Mean |
|---|---|---|---|---|
| Gated (zero-init) | +0.006 | +0.0005 | +0.00004 | **+0.002 (spectator)** |
| Hardcoded 0.5 | +0.419 | +0.523 | +0.433 | **+0.458 (harmful dep.)** |
| Stop-gradient | +3.721 | +346.5 | +15.35 | catastrophic |

### Key findings

1. **Hardcoded 0.5 lateral mixing is harmful** (+0.124 nats vs single-block). The architecture forces block A to consume block B's output at full strength, creating a destructive dependency.

2. **Zero-init gate fixes the ceiling** — gated model nearly matches single-block (1.892 vs 1.882, +0.010). This confirms the hardcoded mixing was the problem.

3. **Block B is a spectator when given the choice.** With zero-init gates, the model keeps gates near zero (ablation effect +0.002 nats). It prefers to operate as a single-block model.

4. **Stop-gradient is catastrophically broken** on the hardcoded architecture (val_loss 2.8–12.8). Not tested on gated architecture because the gated ceiling shows B adds nothing anyway.

5. **The local learning question is unanswerable at this scale.** Block B provides no value even with full backprop + the option to use it. The task (TinyShakespeare ctx=32) is too easy for 1 block — a second block has nothing useful to add.

---

## Diagnosis: task too easy for 2 blocks at this scale

The zero-init gate experiment is definitive: **even with full backprop and the freedom to use block B, the model chooses not to.** Block B adds nothing at TinyShakespeare ctx=32 with d_model=72.

The prior "hardcoded 0.5" failure was a red herring for the local-learning question — it was an interface bug, not evidence about whether blocks can learn locally. Fixing the interface reveals the deeper issue: the task simply doesn't need a second block.

**Why the task is too easy:** The single-block `ParallelDiagonalModel` (no attention, just FFN + token mixing per step) achieves val_loss 1.882 with 53K params. This is already within 0.24 nats of the transformer baseline (1.643 with 186K params). A second block in this architecture can only provide "another perspective" on information block A already has — and at ctx=32 with this simple task, there's no additional perspective needed.

**What this does NOT mean:**
- NOT "propagation delay is broken" (prior evidence: stale reads cost +0.005 nats — negligible)
- NOT "local learning is impossible" (never tested in a regime where the ceiling block helps)
- NOT "multi-block is useless" — see existing WikiText-103 ctx=128 evidence below

---

## Existing WikiText-103 ctx=128 evidence (discovered post-experiment)

`experiments/wikitext_103/artifacts/ctx128_corrected/` contains a prior run at the larger scale with the hardcoded 0.5 mixing:

| Config | Val loss (2-seed mean) | Δ vs single |
|--------|----------------------|-------------|
| A_single (1-block, 2.8M) | ~1.84 | — |
| B_corrected (4-block, 0.5 mixing, token_injection=block0) | ~1.86 | **+0.014 worse** |
| C_old (4-block, token_injection=all) | ~1.82 | **-0.021 better** |

**Key insight:** The spectator problem persists at WikiText-103 ctx=128 with hardcoded 0.5 mixing. But C_old proves extra blocks CAN help when they receive fresh tokens directly. The question is whether the zero-init gate (which fixes the interface at TinyShakespeare) also fixes it at this scale.

---

## Gated multi-block at WikiText-103 ctx=128 (2026-05-25)

**Result: B_gated is 0.245 nats WORSE than A_single. Pathway 3 remains blocked.**

| Variant | Params | Final val_loss | Δ vs single | Wall time |
|---------|--------|---------------|-------------|-----------|
| A_single (1-block) | 2.85M | 1.832 | — | 469s |
| B_gated (4-block, zero-init gates) | 3.64M | 2.077 | **+0.245 worse** | 721s |

Gate values at end of training: `[-0.003, -0.009, -0.115]`

| Block | Ablation effect (nats) | Interpretation |
|-------|----------------------|----------------|
| 0 | +3.41 | Load-bearing (has tokens + readout) |
| 1 | +0.001 | Spectator |
| 2 | +0.002 | Spectator |
| 3 | +0.056 | Marginally useful (gate opened negatively) |

### Diagnosis: cold-start problem

The zero-init gate creates a **chicken-and-egg** optimization failure:
- Gate starts at 0 → blocks 1-3 receive NO lateral signal from block 0
- Without signal, blocks 1-3 can't develop useful representations
- Without useful representations, gradient through the gate is noisy
- Gates stay near zero → blocks remain information-starved
- `readout_mode="all"` averages these useless states with block 0's good state → overall performance degrades

Only gate 3 (rate=8 block, fires every 8 steps) opened to -0.115. The **negative** value means block 3 learned to SUBTRACT block 2's noisy output — a suppressive function, not constructive lateral use.

### Why this is WORSE than B_corrected (hardcoded 0.5, val_loss ~1.86)

B_corrected forces `0.5*(state + neighbor)`, which at least gives blocks 1-3 half the signal from below. They have something to compute on. Zero-init gates give them NOTHING — creating a strictly worse cold-start than forced mixing.

### What this does NOT prove

- ❌ Does NOT prove "lateral-only blocks are fundamentally useless" — the cold-start confound makes the result uninformative about this
- ❌ Does NOT prove "the zero-init gate is a bad interface in general" — it's specifically bad when combined with no-token blocks that start with no signal
- ❌ Does NOT answer the local learning question — still premature

### What it DOES prove

- ✅ Zero-init gates + upward topology + block0-only tokens creates a degenerate initialization
- ✅ The readout confound (reading from useless blocks) actively harms performance
- ✅ Gate 3 opening negatively shows the model CAN learn gates — but only to suppress, not to use constructively
- ✅ The "fix the interface" hypothesis was incomplete: the problem isn't just the mixing coefficient, it's the initialization + information routing

---

## What this experiment will NOT settle

- Whether local learning scales beyond 2 blocks
- What the optimal local signal is
- Whether the architecture works at larger scale / longer context
- The gradient radius sweep (depends on blocks being useful first)
- ✅ ~~Whether the 0.5-averaging topology is optimal~~ → **ANSWERED: 0.5 is bad but zero is worse (cold-start)**

---

## Exit conditions (revised after gated experiment)

**Resolved:**
- ✅ Hardcoded 0.5 lateral mixing is harmful at TinyShakespeare (+0.124 nats)
- ✅ Block B is a spectator at TinyShakespeare ctx=32 even with full backprop
- ✅ The local learning question is premature — no regime exists where lateral-only blocks help
- ✅ Zero-init gate creates cold-start problem at WikiText-103 ctx=128 (+0.245 nats vs single-block)
- ✅ Only token_injection=all (C_old) makes multi-block useful at this scale (-0.021 nats)

**Open:**
- ❓ Does C_old's improvement come from lateral communication, or is it just an ensemble of independent token-fed blocks? (Eval ablation needed)
- ❓ Can local learning work in C_old config (where blocks ARE useful)?
- ❓ Is there a gate initialization that avoids the cold-start problem while still being learnable?
- ❓ Is the "upward" topology (no feedback from upper to lower) the right choice?

---

## Prior relevant evidence

| What | Result | Implication |
|---|---|---|
| Stale-read quality cost | +0.005 ± 0.005 nats | Delay itself is not harmful |
| Strict-local (no propagation delay) | Collapsed (+1.38 nats) | Wrong architecture — no baseline comparison |
| Neighborhood-local CE (no propagation delay) | Marginal (-0.006 nats) | Slightly helpful but wrong architecture |
| Multi-rate [1,2,4,8] spectator problem | Block 0 sees all tokens → no specialization | Information routing matters |
| Tied-depth iteration scaling (1 block) | Quality peaks ~8 iterations | 1 block CAN do a lot — task may be too easy for 2 blocks at this scale |

---

## Artifacts

- Training logs: `experiments/propagation-delay/artifacts.ignore/*.jsonl`
- Experiment script: `experiments/propagation-delay/run.py`
