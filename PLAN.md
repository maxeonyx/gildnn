# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current operational state (2026-05-26, 11:10 NZST)

**Warmup→detach RUNNING** (PID 20952, launched 10:13 NZST). ~6 runs × 26 min. ETA ~12:50 NZST.
- Monitor: `Get-Content "experiments/wikitext_103/artifacts/warmup_detach/run.jsonl" -Tail 3`
- Seed 42 complete (both conditions). Seed 43 warm12_detach in progress (step 5K/20K as of 11:10).

**Daily report 2026-05-26:** NOT YET WRITTEN. Due after 4pm NZST.

### GPU queue

| # | Experiment | Status | Duration | Notes |
|---|---|---|---|---|
| 1 | ~~bridge_detach~~ | **DONE** | — | 3 seeds concordant |
| 2 | **warmup_detach** | **RUNNING** | ~2.5h | ETA ~12:50 |
| 3 | tied_sharing rerun | Queued | ~45 min | `runs/tied_sharing.py`, sanity-checked ✅ |
| 4 | predictive-residual | Conditional | ~2h | Only if warmup confirms Scenario A (3 seeds) |

---

## Decision state (what's been decided)

| Decision | Outcome | Constraint it imposes |
|---|---|---|
| Temporal_window 2-block | Branch 1 confirmed (Δ=+0.042, 3 seeds) | Trajectory info uniquely helps intended arch |
| 4-block follow-up | **Stop-loss fired** (conditions 4+5 fail) | **No more intended-architecture rescue experiments unless Max overrides** |
| Bridge_detach | Clearly worse (gap +0.027, 3 seeds) | Gradient IS needed for block specialization |
| Surrogate vs intended | Surrogate results useful for mechanism isolation | Do NOT let surrogate findings silently become "the architecture" |
| Timebox | ~5 days remaining | Budget ~3-4 experiments after tied_sharing |

**Traps to avoid:** more intended-architecture rescue (stop-loss fired), custom CUDA kernels (infra not findings), starting new pathways (4/6/7/9/10/11 can't meaningfully start in 5 days).

---

## Active experiment context

### Warmup→detach (RUNNING)

Tests: "Is lateral gradient needed CONTINUOUSLY or just for bootstrapping?"

**Seeds 42-43 results (warm12 complete for both; warm15 seed 43 in progress):**

| Seed | Condition | Final val_loss | R | Gap re-opening | Interpretation |
|---|---|---|---|---|---|
| 42 | warm12_detach | 1.7977 | -0.14 | +0.033 | No recovery |
| 42 | warm15_detach | 1.7863 | 0.26 | +0.021 (>0.010) | Temporary head start only |
| 43 | warm12_detach | 1.7925 | 0.03 | +0.037 | No recovery |
| 43 | warm15_detach | *running* | — | — | — |

Baselines (from bridge_detach per seed): seed 42 full=1.7650 det=1.7936; seed 43 full=1.7558 det=1.7935.
Both seeds concordant: drifted back toward detached → **Scenario A (gradient needed continuously) confirmed (2/3 seeds).**

Pre-registration and interpretation framework: `research/questions/local-learning/README.md` lines 477-530.

### Bridge_detach (COMPLETE — key reference for warmup)

3 seeds concordant "clearly worse" (mean gap +0.027 nats).

**Readout pattern (the discriminator):**
- Full_backprop: "U-shaped" — block 3 contributes +0.35-0.42 (all seeds)
- Detached: "front-loaded" — block 3 contributes +0.07 (all seeds), block 1 compensates (+0.41-0.68)
- Interpretation: without lateral gradient, senders don't learn what to send → receivers can't specialize

Full artifact: `experiments/wikitext_103/artifacts/bridge_detach/`

### Tied_sharing (QUEUED)

Clean weight-sharing isolation: shared vs distinct feedforward, both with token_injection=all. Prior data was LOST during refactoring. Provisional 1-seed result: tied_shared=1.910 vs distinct≈1.85 (~0.06 gap). Needs 3-seed rerun.

**Honest framing:** This shares only feedforward weights — per-block `token_mixes` and `block_mixes` remain distinct. Tests "shared feedforward processing is viable with position-specific routing," NOT full Pathway 1 hypothesis.

### Predictive-residual (CONDITIONAL — next after tied_sharing)

Pre-registered: `research/questions/local-learning/README.md` lines 532-583.
- Adversarial-reviewed ✅
- Shuffled-target control (tests whether prediction itself helps vs just the gradient)
- Implementation: model has `forward_with_state()` for per-block outputs. Needs custom GraphTrainer (standard one hardcodes `model()` + `F.cross_entropy`; this needs `forward_with_state()` + aux loss inside the captured graph).
- Template: `runs/bridge_detach.py` (custom training loop, NOT `run_training_loop` from core/)

---

## Active gaps / open evidence

- **Clean weight-sharing isolation** — data lost, rerun queued (tied_sharing). Independent core question.
- **Predictive-residual local loss** — pre-registered, awaiting warmup confirmation + tied_sharing completion.
- **No clean dynamic-depth PREDICTOR** — oracle speedup is 1.96×, but "can shallow state predict optimal depth?" untested.
- **TinyShakespeare ceiling** — every experiment there saturates at 20-40K steps. WikiText-103 is the valid testbed.

---

## The agent's working loop

Follow PROCESS.md. The short version:

```
PROCESS CHECK → PATHWAY SELECTION → [adversarial gate] →
CONCEPTUAL CLARIFICATION → [adversarial gate] →
EXPERIMENT DESIGN → [adversarial gate] →
RUN & ANALYZE → [adversarial gate] →
INTEGRATE / REPORT / UPDATE PLAN → back to PATHWAY SELECTION
```

Every gate is a separate subagent review that can send you back. See PROCESS.md for full details.

---

## Completed work (reference)

| What | Pathway | Result | Key finding |
|---|---|---|---|
| Transformer baseline (TS) | all | val_loss 1.643, 186K params | Done, in base-experiments/ |
| RNN baseline (TS) | all | val_loss 1.711, 186K params | Done, in base-experiments/ |
| **Transformer baseline (WT-103)** | **all** | **1.592 ± 0.003, 2.86M params** | **External anchor. 0.240 nats better than A_single.** |
| **Tied-depth tiny rung** | **1** | **Mean diff 0.0002, 3 seeds** | **Weight sharing free at tiny scale** |
| **Tied-depth WT-103** | **1** | **Seed-sensitive, confounded** | **Cannot isolate weight sharing (token_injection differs). Clean test needed.** |
| **Dynamic-depth oracle** | **5** | **Speedup 1.96×, 71.5% harmed by d=8** | **Worth pursuing. Pathway 5 alive.** |
| **Temporal_window 2-block** | **3** | **Δ_trajectory=+0.042, 3 seeds** | **Branch 1: trajectory uniquely helps intended arch** |
| **4-block follow-up** | **3** | **Stop-loss fires** | **Block 1 rescues (6×), blocks 2-3 stay spectators** |
| **C_old lateral ablation** | **3** | **Δ=+0.030, 2 seeds** | **Laterals load-bearing. All 4 blocks contribute.** |
| **Bridge_detach** | **3** | **Clearly worse, +0.027, 3 seeds** | **Gradient needed for specialization. U-shaped vs front-loaded readout.** |
| Multi-rate [1,2,4,8] | 8 | Spectator problem | Block 0 sees all tokens → no specialization |
| CUDA graph training | infra | 10.8× speedup | In core/ |
| CUDA graph concurrency | 2 | 28% speedup | Hardware CAN do concurrent execution |
| Stale-read quality | 2 | +0.005 ± 0.005 nats | Negligible |

---

## Parked / future strands

### Fading strands (from first-principles re-derivation, 2026-05-25)

Not actionable now, but represent significant vision chunks:

1. **Attention-based routing** — original note says "how do inputs get aggregated? They use attention." Current arch uses fixed mix-add. Revisit once intended arch works.
2. **Loss prediction as decision mechanism** — "a prediction head predicting the loss of another." Central to Pathways 4/5/6. Needs working architecture first.
3. **Graph topology** — Max's vision is a graph, not a stack. Major unexplored dimension. Correctly deprioritized.
4. **Hierarchical dynamic tokenization** (Pathway 7) — stacked autoencoders with learned chunk boundaries. Needs working base architecture.

### Multi-timestep architecture (dictation 2026-05-25-1)

Max's LATEST design thinking evolves beyond current ParallelDiagonalModel. Full write-up: `research/questions/multi-timestep-architecture/README.md`.

Key departures:
- Streams carry **distributions** (diagonal Gaussians: μ, σ), not point vectors
- Local loss = **Wasserstein distance** (block predicts left neighbor's next distribution)
- No cross-block gradients needed by design
- Blocks are temporal edges on a 2D grid (lateral positions × timesteps)
- Combining function is shared/tied across all positions

**Connection to current work:** If warmup→detach confirms gradient needed continuously (Scenario A), the Wasserstein local loss is the natural escalation — provides a RICHER local signal (predict left neighbor) rather than relying on weak shared adjoint alone.
