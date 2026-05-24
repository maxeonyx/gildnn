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
| 2-block full backprop | 2.008 | 1.963 | 2.048 | **2.006** | 99K |
| 2-block stop-gradient | 2.820 | 4.889 | 12.756 | **6.822** | 99K |

### Batch-shuffle ablation (block B load-bearing test)

| Condition | Seed 42 | Seed 137 | Seed 2024 | Mean |
|---|---|---|---|---|
| Full backprop | +0.419 | +0.523 | +0.433 | **+0.458** |
| Stop-gradient | +3.721 | +346.5 | +15.35 | catastrophic |

### Key findings

1. **2-block full backprop is WORSE than single-block** (+0.124 nats mean), despite having 2× parameters. The architecture creates a harmful dependency — block B is heavily load-bearing (ablation +0.458) but the joint system lands in a worse optimum than not having block B at all.

2. **Stop-gradient is catastrophically broken.** Val losses of 4.89–12.76 (some above random chance ≈ log(65) ≈ 4.17). Massive seed variance. Block B learns representations for its own local CE that actively harm block A.

3. **Block B is NOT a spectator** — it's heavily used in both conditions. The problem is not information flow; it's that the architecture forces harmful coupling.

---

## Diagnosis: hardcoded 0.5 lateral mixing

The root cause is `block_input = 0.5 * (state_input + neighbor_state)` in `ParallelDiagonalModel`. This hardcoded average:

- Forces block A to consume block B's lateral output at 50% weight with no way to down-weight it
- Attenuates block A's own useful signal (token embedding) to 50%
- Creates mandatory coupling: even random/harmful lateral states directly corrupt the receiver

Under full backprop: blocks co-adapt around this constraint, landing in a worse joint optimum. Under stop-gradient: block B optimizes for its own CE, producing representations that are actively harmful to A, and A has no mechanism to ignore them.

**What this is NOT:** evidence that propagation delay is fundamentally broken. Prior evidence (stale-read cost +0.005 ± 0.005) showed delay is benign. The problem here is the mandatory coupling interface, not the temporal delay.

---

## Next step: zero-init learnable lateral gate

Replace the hardcoded `0.5 * (self + neighbor)` with:

```
block_input = state_input + g * neighbor_state
```

Where `g` is a scalar parameter initialized to 0. This means:
- At initialization, the model behaves like single-block (lateral ignored)
- The model discovers how much lateral to use via gradient
- If lateral input is harmful, g stays near 0

**Success criteria:** Full-backprop 2-block with zero-init gate achieves val_loss ≤ 1.882 (matching single-block). Ablation effect > 0.02 (B actually contributes). If this works, rerun stop-gradient on the gated architecture.

**If this also fails:** try unidirectional topology (`topology="upward"`, `readout_mode="last"`) — removes the bidirectional feedback loop.

---

## What this experiment will NOT settle

- Whether local learning scales beyond 2 blocks
- What the optimal local signal is (this tests only local CE — one candidate)
- Whether the architecture works at larger scale / longer context
- The gradient radius sweep (that's ROADMAP Step 2, depends on this result)
- Whether the 0.5-averaging topology is optimal ← **ANSWERED: it's bad**

---

## Exit conditions (revised)

The original H1/H2/H3 framework was premature — the experiment revealed an architecture design flaw before the local-learning question could be answered.

**New path:**
1. Fix the ceiling (zero-init gate) → does 2-block full-backprop match or beat 1-block?
2. If yes → rerun stop-gradient on the fixed architecture → does local CE work now?
3. If still no → the interface problem is not just the mixing strength; try other topologies/signals

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
