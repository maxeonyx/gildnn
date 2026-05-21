# Residual Streams Across Time

**Question:** What does "residual stream across time" mean as an operational architecture — and what is the smallest faithful experiment that would test it?

**Status:** Theory-first analysis. No experiment has been run yet that faithfully instantiates this concept. This report exists to make the concept precise enough to build it correctly.

**Grounded in:** [dictations/2026-05-20-15.md](../../../dictations/2026-05-20-15.md), [dictations/2026-05-20-16.md](../../../dictations/2026-05-20-16.md), [dictations/2026-05-20-10.md](../../../dictations/2026-05-20-10.md)

---

## The core insight

In a standard transformer, residual streams run across depth. Each layer reads the stream, adds something to it, and passes it along. The stream at any point is a superposition of contributions from all previous layers. Nothing is erased. Nothing is gated. It just accumulates.

Max's idea is to take this structure and apply it in the time direction.

From [2026-05-20-10](../../../dictations/2026-05-20-10.md):

> *"Let's remove the recurrent part of this. I don't want recurrent blocks, I want regular residual blocks. And I'm imagining that the boundary between the nodes is essentially a residual stream. It's not a separate size, it is D model."*

And critically:

> *"If you unroll across time, then putting stop gradients across time is also what I'm thinking about, by the way. And using the residual connections across time, especially if we're doing attention residuals, it's essentially the same thing as doing a transformer. That is, a transformer with attention residual connections, but with looped blocks."*

This is the key grounding analogy: **a residual stream across time is a transformer where the sequence dimension is time, not the input tokens, and where the same block (or blocks) is reused at each timestep**. The stream at timestep T is a D-dimensional vector that represents the accumulated state of what has been processed so far.

---

## What persists across timesteps — and what does not

### The residual stream

At each timestep T, there is a vector `s_T ∈ ℝ^{d_model}`. This is the residual stream at time T. It is *not* an RNN hidden state in the traditional sense. The differences are worth being explicit about:

| | RNN hidden state | Residual stream across time |
|---|---|---|
| Update rule | `h_T = gate(h_{T-1}, x_T)` | `s_T = s_{T-1} + f(s_{T-1}, x_T)` |
| Forgetting | Explicit (gating) | No — additions only accumulate |
| Norm behavior | Bounded by gate | Can grow; needs separate norm control |
| Gradient flow | Through gates (vanishing/exploding issue) | Through residual add; stop-gradient is explicit choice |
| Block output | New state that replaces the old | Delta added to the persistent stream |

The residual stream is not wiped between timesteps. Each block contributes an additive update. The GRU `reset_gate * h_{T-1}` mechanism is specifically what Max is *against* — that is gating, not addition.

### Block state vs timestep state

With a **single block reused across time** (the simplest case Max points toward):

- **The stream** `s_T` is the full state at time T. It has dimension `d_model`, same as any other block boundary.
- **The block itself** has no separate per-instance state — it is just a function `f: ℝ^{d_model} → ℝ^{d_model}` applied at each step.
- **The history** of the stream at prior timesteps `{s_0, s_1, ..., s_{T-1}}` is what temporal coupling mechanisms can optionally read.

With **multiple blocks** and diagonal connections, the picture gets richer (see Design Tree below), but the primitive is the same: a uniform-width stream where blocks add rather than replace.

---

## How this differs from a standard RNN

The standard RNN intuition is: the hidden state `h_T` *summarizes everything up to T*. A GRU has sigmoid-gated forgetting to stop the state from exploding and to let it track long-range dependencies. The hidden dimension is often different from the input dimension.

The residual-stream intuition is different: the stream is a *superposition*, not a summary. Nothing is forgotten by design; instead, you rely on the linearity of addition and on any attention mechanism to pick out what's relevant. The block's job is not to compress the past — it's to add something useful to a common channel that everything can read and write to.

This is why Max keeps coming back to the transformer analogy. In a transformer, the residual stream after layer L is a superposition of contributions from all previous layers. No layer is "in charge" of the full state. The same principle extends to the time axis.

---

## Design tree of architectural choices

The following branches cover the choices Max named across the dictations. This is not necessarily complete — it maps the space as he described it, not as an exhaustive enumeration. Not all combinations are sensible — pruning reasoning follows each branch.

### 1. Temporal coupling mechanism

How does information from past timesteps reach the computation at the current timestep?

**A. Plain residual add**
The block at time T simply takes `s_{T-1}` as input and adds its output back:
```
s_T = s_{T-1} + block(s_{T-1}, x_T)
```
This is the simplest possible temporal coupling. The block can in principle "remember" the past by encoding relevant information in `s_{T-1}` — it just has to rely on its own outputs from prior steps. This is the control: does basic temporal residual connection even do anything useful?

**B. Learned mix-add over the stream**
Replace the `+` with the mix-add operation from [mix-add](../mix-add/) ([dictation reference](../../../dictations/2025-05-08-1.md#dictation-5)):
```
s_T = mix(s_{T-1}, block(s_{T-1}, x_T), m)
```
where `mix(a, b, m) = a * sqrt(σ(m)) + b * sqrt(1 - σ(m))`. This is approximately norm-preserving when both inputs are unit-norm. Whether it helps training stability here — where the stream is accumulated over many timesteps — is an open question. It is a candidate to compare against plain add.

**C. Causal attention over past residual states**
The block at time T has access to a short history `[s_{T-k}, ..., s_{T-1}]` and computes attention-weighted reads over those states, then adds to the stream:
```
context = CausalAttn([s_{T-k}, ..., s_{T-1}])
s_T = s_{T-1} + block(s_{T-1}, x_T, context)
```
Max says he is *"interested in attention over the past"* ([2026-05-20-15](../../../dictations/2026-05-20-15.md)) — the same dictation also names mix-add as a candidate. Both are things he wants to explore; neither is declared preferred over the other. The window length `k` is a design choice — short window is cheap and testable; long window approaches full sequence attention over the temporal dimension.

**D. Attention + mix-add**
Combine C and B: attention over past states, with mix-add combining the result into the stream. This is probably not the first thing to test — it conflates two variables.

**Pruned: gating mechanisms (GRU, LSTM)**
Max is explicit: *"I don't like these gating mechanisms, I like residual streams."* The GRU reset/update gate is a forgetting mechanism. It is not compatible with the accumulation primitive Max wants to explore. These are excluded from the search space regardless of their empirical performance.

### 2. Residual topology

How do blocks connect across time?

**B. Diagonal block coupling** *(what Max named explicitly)*
From [2026-05-20-15](../../../dictations/2026-05-20-15.md): *"Going from residual block A to the next block in the next time step, a diagonal step."*

The diagonal is block-to-*next-block* across time, not same-block across time. In a multi-block arrangement (blocks B1, B2, B3 running in sequence within each timestep), block Bi at time T receives a residual contribution from block B_{i-1} at time T-1 — one step back in time, one step up in depth. That is the diagonal: `(depth i-1, time T-1) → (depth i, time T)`.

Visually, in the (depth × time) grid:

```
       T=0    T=1    T=2
B3:    [  ]   [  ]<\ [  ]
        ↑      ↑ \  ↑ \
B2:    [  ]   [  ]<\ [  ]
        ↑      ↑ \  ↑ \
B1:    [  ]   [  ]  [  ]
```

Vertical arrows (↑) are depth connections within a timestep. Diagonal arrows (\) are the new cross-time residuals: B1(T-1) → B2(T), B2(T-1) → B3(T), etc. This is distinct from a horizontal connection (same block across time: B1(T-1) → B1(T)) and from the vertical connection (previous depth at same time: B_{i-1}(T) → Bi(T)).

The operational implication: each block in the stack has two incoming residual streams — one from below it at the current timestep, one diagonally from the block below-and-back. These combine before (or as part of) the block's computation. The gradient flow is also diagonal: the diagonal connection must be covered by the stop-gradient choice.

**A. Same block across time (looped / horizontal)**
One block `f` reused at every timestep. The same weights process the stream at T=0, T=1, T=2, .... This is weight-tied temporal recurrence without gating. It relates to the grounding analogy Max gave — *"a transformer with attention residual connections, but with looped blocks"* — but the connection here is horizontal (same block), not diagonal (next block). This is the control topology: it uses the correct primitive (residual add, no gates) without yet adding the diagonal structure Max named as "the main thing."

**C. Both**
Full topology: diagonal residual connections between adjacent blocks across time, plus standard depth-wise residuals, plus same-block temporal coupling if desired. This is the richest version and the hardest to build cleanly as a first probe.

**What the topology choice implies for the minimal probe:** A (looped single block) is the tractable starting point — it tests the temporal residual primitive without multi-block machinery. B (diagonal) is what Max named as *"the main thing I really want to explore"* — it needs at least two blocks and the diagonal connection wired explicitly. These should be tested in sequence, not combined with other variables on the first run.

### 3. Gradient coupling

Whether and where backpropagation flows across timestep boundaries.

**A. Fully end-to-end**
No stop-gradients. Gradients flow through the temporal residual connection across all timesteps. This is the baseline: does the architecture learn at all?

**B. Stop-gradient at every timestep boundary**
`s_{T-1}` is treated as a constant when computing gradients for time T. Each block only updates based on local loss at T. This is what enables async execution to be meaningful for training: if block B1's gradient at T=5 doesn't depend on B2's activations at T=4, B2 doesn't have to finish T=4 before B1 can start T=5's backward pass. Max explicitly connects stop-gradients to async performance in [2026-05-20-15](../../../dictations/2026-05-20-15.md): *"I also do expect stop gradients to be important for async performance in training."*

**C. Stop-gradient at selected boundaries only**
Partial decoupling: detach across certain time boundaries, keep full gradients elsewhere. This is probably not the first thing to explore — it adds a hyperparameter without a clear principled choice for where to put the boundaries.

**Max's expectation:** stop-gradients are probably needed for async throughput gains — he says *"I also do expect stop gradients to be important for async performance in training."* That's an expectation, not a confirmed result. Whether stop-gradients help or hurt training *quality* is a separate and unknown question. These should be measured independently.

### 4. Async execution semantics

What "async" actually means in practice, at training time.

**Sync approximation (training-time default):** All blocks run in lockstep. Timestep T completes fully before T+1 begins. Temporal residual connections use fresh values. This is the tractable starting point. Async is a future properties property, not a training-time requirement.

**Stale-read approximation:** Blocks read from shared memory that may not be the most recent value — it's whatever was last written by a neighboring module. This is the mechanism Max describes: *"we're going to simply pack more into a certain area of the GPU, have one area of the GPU rapidly iterate on one particular recurrent module unit."* In practice this means some blocks run more iterations per wallclock unit than their neighbors, producing temporal desynchronization. The neighbor reads stale values but doesn't block.

**True desynchronized schedule:** Different blocks run at genuinely different rates, with no synchronized "timestep" at all. Block B1 might be at T=100 while B3 is at T=95. This is the full vision Max describes — time compression/dilation as an emergent property, not a hardcoded schedule. It requires stop-gradients (otherwise the backward pass needs synchronization) and is probably only tractable after the synchronous architecture is working.

**Key point:** async is a *systems property* Max expects to improve FLOPs/second. He explicitly expects it to hurt accuracy somewhat. The correctness of the architecture is independent of whether async is enabled. These should be tested separately.

### 5. Broadcast/global channel

How do distant blocks communicate.

**Absent (first probe):** No broadcast. Blocks can only read from their immediate temporal history and their local depth neighbors. This isolates the temporal residual coupling question from the communication question.

**Attention-based broadcast:** From [2026-05-20-16](../../../dictations/2026-05-20-16.md): *"the broadcast mechanism I'm imagining is something like attention over all of the modules and then it would broadcast essentially the result of that attention to all modules."* A central router attends over all block states and writes a mixed signal back. It should use stale reads to avoid synchronization overhead (per Max's explicit requirement in [2026-05-20-16](../../../dictations/2026-05-20-16.md)).

**Reward-connected broadcast:** The broadcast mechanism receives an auxiliary supervised or RL signal that steers the mixture. Max speculates this may be what enables the broadcast to learn anything useful: *"Without it, I think we try without the broadcast. That's my guess. Because it's probably can't learn to do anything useful with local learning."* This is the longest-range design choice and is firmly not in scope for the first probe.

**What is open:** Whether the broadcast can learn useful things with local learning alone is genuinely unsettled. Max is explicitly uncertain. Do not collapse this to a conclusion.

---

## What variants are obviously wrong, and why

**GRU/LSTM gating:** Wrong primitive. Gating forgets; residuals accumulate. The project question is specifically about what accumulation-based temporal coupling does — GRU gives a different and already well-understood answer.

**Mean-pool broadcast (already tested):** The [async-volatile-memory](../async-volatile-memory/) result shows that naive mean pooling across all block outputs doesn't work as a broadcast mechanism. This is not strong evidence against broadcast in general — it's one cheap version that failed. Attention-based broadcast (mechanism 5B above) is still live.

**Unrestricted all-to-all attention over blocks:** If every block can attend to every other block at every timestep, the graph structure becomes decorative and the async execution becomes impossible (you need full synchronization to gather inputs for the attention). This is noted in [VISION.md](../../../VISION.md) as a risk: *"unrestricted all-to-all attention between blocks may bypass the interesting parts entirely."* The causal window attention (mechanism 1C) over a *short* history is specifically designed to avoid this.

**Full end-to-end gradients with no stop-gradient:** Not "wrong" as a baseline, but it forecloses async execution and removes the local-learning question entirely. It is only valid as a reference baseline for measuring what stop-gradients cost or gain.

**Mixing multiple speculative mechanisms in the first probe:** Don't combine diagonal topology + attention over past + mix-add + local learning + broadcast in one experiment. None of those pieces have been validated individually in this architecture family. Per the project process ([PROCESS.md](../../../PROCESS.md)), isolation before composition is non-negotiable.

---

## How local learning fits here

Max says in [2026-05-20-15](../../../dictations/2026-05-20-15.md): *"I still don't understand the local learning. We haven't explored this enough."* And: *"What exactly do these residual blocks look like so that their local loss is such that they learn higher level representations of stuff based on their distance away from the input stream?"*

This is an open question, not a settled design choice. The provisional concept is:

Each block at each timestep has a local predictive loss — it tries to predict the value of the residual stream it will receive at the next timestep. This is self-supervised at the block boundary. The block is not trying to predict the final model output; it is trying to predict its own future input. The hypothesis is that this pressure causes blocks to encode progressively more abstract or slowly-varying representations as you go deeper (further from the raw input stream).

Whether this works in a residual-stream-across-time architecture specifically is unknown. The same concept was tried in a simplified form in [local-learning-residual](../local-learning-residual/) with inconclusive results — but that was on a GRU-based architecture, which changes the dynamics significantly.

**Open questions for local learning in this architecture:**
- Does the self-prediction objective produce meaningful gradients when block outputs are residually added (rather than replacing the state)?
- Does depth (block index) still create a gradient-decoupled hierarchy when the stream passes through all blocks at every timestep?
- Can local learning substitute for global end-to-end gradients, or does it collapse to a trivial prediction strategy (e.g., always predicting `s_{T-1}` unchanged)?

These questions are real and should not be experimented on before the base architecture (temporal coupling without local learning) is understood.

---

## The broadcast mechanism in this frame

Max's description from [2026-05-20-16](../../../dictations/2026-05-20-16.md) gives the clearest operational picture:

1. **Stale reads are mandatory.** The broadcast mechanism must read from all modules without synchronizing them. If it waits for every block to finish before reading, it serializes the entire network and kills throughput. Stale reads are not an approximation to be fixed later — they are the mechanism that makes the broadcast compatible with async execution.

2. **Async writes back.** Modules read the broadcast output with the same stale tolerance they apply to their own temporal stream. The broadcast writes when it has something to say; modules read whenever they next compute.

3. **Attention over module states.** The broadcast is not a simple average. It attends over block states to compute a weighted mixture, potentially routing information from a specific block to all others. How it learns what's useful is open.

4. **Tentative: reward signal needed.** Max speculates that without a reward signal (RL or supervised), the broadcast mechanism probably can't learn what to route: *"I'm not 100% sure how that should work. I don't think that in the human brain the attention is purely predictive."* This makes the broadcast a later-phase addition, not a first-probe component.

**Implication for the probe:** exclude the broadcast. Its useful operation is probably conditional on a reward signal, and that signal introduces another variable. Test the temporal residual coupling first.

---

## Proposed minimal faithful probe

One concrete architecture. Not a family of experiments — one set of choices from the design tree, chosen to be the smallest thing that honestly instantiates the concept.

### The probe

A character-level language model (Karpathy dataset) with:

- **One residual block** `B` with standard transformer-style internals (LayerNorm → self-attention → FFN → residual add). Input and output are both `d_model` — no separate hidden dimension.
- **Unrolled across time.** At each timestep T, the block reads the residual stream from the prior step and the input embedding, then adds its output:
  ```
  s_T = s_{T-1} + B(s_{T-1}, x_T)
  ```
  No gating. No separate hidden state. Plain residual add.
- **Causal attention over a short window of past residual states** `[s_{T-k}, ..., s_{T-1}]` inside the block, with window `k=4`. This is one of the two temporal coupling mechanisms Max named; it is included because it is what makes this different from a trivial looped FFN.
- **Output head:** linear projection from `s_T` to vocabulary logits, cross-entropy loss vs next character.
- **No broadcast. No local learning. No stop-gradient. No async execution.** All deferred.

That is the probe. One block, one temporal coupling mechanism, plain residual add, global end-to-end gradients.

### Why these choices

- Single block keeps the architecture legible and the failure modes interpretable.
- Plain residual add (not mix-add) as the temporal combinator avoids confounding the coupling question with the norm-preservation question.
- Window attention `k=4` over past states rather than `k=0` (plain residual) because without it the probe is just a looped FFN — it doesn't test the attention-over-past mechanism Max named. `k=4` is small enough to be fast and large enough for attention to do something.
- Global gradients as the baseline: establishing that the architecture learns at all before adding the stop-gradient complication.

### Future comparison arms (not part of this probe)

Once the base probe trains cleanly:

- `k=0` ablation: plain residual add with no attention over past — isolates the contribution of the temporal attention.
- Mix-add at the temporal boundary instead of plain `+`.
- Stop-gradient at `detach(s_{T-1})` — measures the quality cost of the async-enabling change.
- Multi-block with diagonal residual connections — the topology Max said is "the main thing."
- GRU control on the same task and parameter count — not the goal, but a reference point for whether the primitive is viable.

### What this probe can tell us

- Can this architecture learn character-level language at all?
- Does the short-window temporal attention produce lower loss than the `k=0` ablation (when that runs later)?
- Is training stable under plain residual accumulation without norm control?

### What this probe cannot tell us

- Whether diagonal block coupling (mechanism 2B) helps — needs multi-block architecture.
- Whether mix-add helps at the temporal boundary.
- Whether stop-gradient degrades loss and by how much.
- Whether local learning or the broadcast mechanism are viable.
- Whether true async execution gives throughput gains.

---

## Experimental results: minimal probe

The theory probe above has now been run in the smallest frame that stayed legible. The tested model is a **single reused residual block** with `d_model=192`, causal attention over the previous `k=4` residual states, plain residual add, and one **shared temporal pre-norm** applied to the stream at the start of each timestep before the FFN block and attention query. Artifacts live under [`experiments/residual_stream_time/`](../../../experiments/residual_stream_time/), with the key runs in [`artifacts/overfit-fast/`](../../../experiments/residual_stream_time/artifacts/overfit-fast/), [`artifacts/scaleup-192/`](../../../experiments/residual_stream_time/artifacts/scaleup-192/), [`artifacts/overfit-timestep-norm/`](../../../experiments/residual_stream_time/artifacts/overfit-timestep-norm/), and [`artifacts/scaleup-192-timestep-norm/`](../../../experiments/residual_stream_time/artifacts/scaleup-192-timestep-norm/).

The most important result is that **temporal pre-norm appears essential in this architecture family**. Without it, the short scale-up run stayed finite but the residual stream norm grew to `1555.07` by the end of the prompt trace, the attention weights drifted toward near-uniform reads over the 4-state window, and generation collapsed to mostly blank continuation ([`scaleup-192/tiny_metrics.json`](../../../experiments/residual_stream_time/artifacts/scaleup-192/tiny_metrics.json), [`scaleup-192/sample.txt`](../../../experiments/residual_stream_time/artifacts/scaleup-192/sample.txt)). An earlier no-pre-norm version was worse again: overfit hit NaNs until attention-side normalization was added ([`overfit-fast/overfit_metrics.json`](../../../experiments/residual_stream_time/artifacts/overfit-fast/overfit_metrics.json)). The current best reading is not "plain add is impossible," but "plain temporal residual accumulation needs a transformer-style read-through norm."

With the added temporal pre-norm, the same reduced scale-up frame (`50K` train chars, `10K` val chars, `3` epochs) reached **validation loss `1.787978` and validation accuracy `47.0%`**, with prompt-trace final stream RMS only `3.33` and clearly non-uniform temporal attention ([`scaleup-192-timestep-norm/tiny_metrics.json`](../../../experiments/residual_stream_time/artifacts/scaleup-192-timestep-norm/tiny_metrics.json)). The generated text is still repetitive, but it is now recognizably English-like:

```text
First Citizen:
Before we proceed any further, hear me speak.

All:
And the prouse the prouse the prouse ...
```

See [`scaleup-192-timestep-norm/sample.txt`](../../../experiments/residual_stream_time/artifacts/scaleup-192-timestep-norm/sample.txt) and [`scaleup-192-timestep-norm/progression_samples.json`](../../../experiments/residual_stream_time/artifacts/scaleup-192-timestep-norm/progression_samples.json).

This is encouraging but still narrow evidence. The current transformer trust anchor on the standardized comparison frame is about **`1.632` validation loss at ~`186K` parameters** ([`PLAN.md`](../../../PLAN.md), [`research/questions/chain-dynamic-depth/README.md`](../chain-dynamic-depth/README.md)), while this residual-stream-time probe reached **`1.788` at `481,019` parameters** — about `2.6x` larger, and also trained on a smaller/shorter `50K` / `3`-epoch budget. That makes the current result a proof-of-viability, not a fair efficiency comparison. The repetitive generation may still be a training-budget problem rather than a decisive architectural failure.

What this probe now tells us is narrower but useful: **the residual-stream-across-time concept can learn at all, temporal pre-norm looks non-negotiable, and attention over past residual states can become selective and useful rather than decorative**. What it does **not** tell us yet is whether this family can be parameter-efficient relative to transformers, whether diagonal cross-time block coupling improves anything, or whether the async version gains systems throughput without unacceptable quality cost.

### Mix-add variant (no LayerNorm)

Per [dictation 2026-05-21-1](../../../dictations/2026-05-21-1.md), Max prefers **learned mix-add** over LayerNorm for residual stream norm management. A variant was tested with all LayerNorm removed and every residual addition replaced by a learned scalar convex combination: `x_new = sigmoid(α) * x + (1 - sigmoid(α)) * delta`. Three per-site learned alphas: token injection, block update, temporal update.

Result at the same frame (d_model=192, 50k chars, 3 epochs): **validation loss `1.790906` at `479,102` parameters** — essentially identical to the LN variant (`1.787978`). Final stream RMS: `1.01` (vs `3.33` with LN, vs `1555` with no norm control). Learned mix values: token `0.49`, block `0.88`, time `0.93` — the model learned to keep ~88-93% of the existing stream on each update, effectively setting its own forgetting rate.

This confirms mix-add is a viable norm-management mechanism. It achieves parity with LayerNorm without any normalization layers. Artifacts: [`experiments/residual_stream_time_mixadd/artifacts/scaleup-192/`](../../../experiments/residual_stream_time_mixadd/artifacts/scaleup-192/).

### Extended training (100k chars, 5 epochs)

The mix-add probe was trained longer to test convergence: `d_model=192`, 100k chars, 5 epochs. Result: **validation loss `1.670509`** at `479K` params — only `0.038` nats from the transformer anchor (`1.632` at `186K` params). The loss was still improving at epoch 5 (trajectory: 1.98 → 1.83 → 1.76 → 1.72 → 1.67), suggesting it has not yet converged.

One training instability was observed: train loss spiked to `51.02` at epoch 4, but recovered by epoch 5 (val loss continued improving). This suggests mix-add prevents catastrophic divergence but doesn't eliminate all instability.

The gap to the transformer (`0.038` nats) at `2.6x` the parameter count means the architecture is **qualitatively viable but not yet parameter-efficient**. Whether this is a fundamental limitation or an optimization/tuning issue remains open. Artifacts: [`experiments/residual_stream_time_mixadd/artifacts/extended-100k-5ep/`](../../../experiments/residual_stream_time_mixadd/artifacts/extended-100k-5ep/).

### Diagonal coupling (2-block variant)

Tested block-1 at time T feeding into block-2 at time T+1, per Max's description of "diagonal residual streams" in [dictation 2026-05-20-15](../../../dictations/2026-05-20-15.md). Two distinct FFN blocks with the diagonal residual connection wired explicitly.

Result: **validation loss `1.787626` at `777,275` parameters** — effectively identical loss to the single-block probe (`1.787978` at `481K`) but at 1.6x the parameter cost. **Inconclusive**: the diagonal connection neither helped nor hurt at this scale/budget. The signal might require more blocks, more training, or more data to emerge. Artifacts: [`experiments/residual_stream_time_diagonal/artifacts/scaleup-192/`](../../../experiments/residual_stream_time_diagonal/artifacts/scaleup-192/).

### Stop-gradient across time

The key async prerequisite: what happens if we detach the stream between timesteps, so no gradient flows backward through time? Each timestep trains only on its own local gradient — the same regime required for true pipelined async execution.

Result: **validation loss `1.893501`** vs full-gradient baseline `1.670509` — a cost of **+0.223 nats**. Loss trajectory: `2.07 → 1.98 → 1.95 → 1.92 → 1.89` (compare baseline: `1.98 → 1.83 → 1.76 → 1.72 → 1.67`). Training was stable (no spikes), overfit still works perfectly, but convergence is slower and the floor is higher.

Runtime was `257s` vs `606s` — **2.36x faster** — because removing backprop-through-time eliminates the sequential gradient dependency. This demonstrates the throughput benefit of detaching, but the quality cost is currently too high to call it free.

**Implications for async:** naive per-timestep stop-gradient is mechanically viable and gives real speed, but the quality hit (+0.223 nats, ~13% relative) is substantial. If async execution requires this detach, it likely needs compensating architectural changes (auxiliary local targets, partial detach every N steps, or better temporal communication) to close the quality gap. Artifacts: [`experiments/residual_stream_time_stopgrad/`](../../../experiments/residual_stream_time_stopgrad/).

### Partial detach (truncated BPTT) sweep

The stop-gradient result raised the question: is the full +0.223 nat cost necessary, or can partial gradient flow recover quality while keeping the speed? Tested detaching every N=2, 4, 8, 16 timesteps — gradient flows within windows of N steps, then is cut.

| Condition | Val loss | Δ from baseline | Runtime | Speed vs full-grad |
|-----------|----------|----------------|---------|-------------------|
| Full gradient | 1.670 | — | 606s | 1.0× |
| N=16 | 1.693 | +0.023 | 321s | 1.89× |
| N=8 | 1.704 | +0.034 | 286s | 2.12× |
| **N=4** | **1.711** | **+0.041** | **246s** | **2.46×** |
| N=2 | 1.762 | +0.092 | 254s | 2.39× |
| N=1 (full stop-grad) | 1.894 | +0.223 | 257s | 2.36× |

**N=4 is the sweet spot.** It recovers 82% of the stop-gradient quality cost (+0.041 vs +0.223) while matching stop-gradient speed (2.46× vs 2.36× — actually slightly faster due to reduced overhead). Beyond N=4, diminishing returns: N=8 adds only 0.007 more quality for 16% more runtime.

The quality curve has a sharp knee: going from N=1 to N=2 recovers 0.131 nats. N=2 to N=4 recovers another 0.051. N=4 to N=8 only 0.007. Most of the gradient signal flows through 4 timesteps — matching the temporal attention window (k=4).

**For async pipelining:** this means you can break gradient flow every 4 steps with minimal quality cost. A pipeline with 4-step stages would get 2.46× training throughput at only +0.041 nats quality penalty. Artifacts: [`experiments/residual_stream_time_partial_detach/artifacts/sweep/`](../../../experiments/residual_stream_time_partial_detach/artifacts/sweep/).

### Parameter-matched comparison (185K params)

The d_model=192 probe uses 479K params — 2.6× the transformer anchor's 186K. To test whether the architecture is fundamentally less parameter-efficient (vs just benefiting from extra capacity), tested at d_model=116 giving 184K params.

| Model | Params | Val loss | Gap |
|-------|--------|----------|-----|
| Transformer anchor | 186K | 1.632 | — |
| **Residual-stream-time (d=116)** | **184K** | **1.717** | **+0.085** |
| Residual-stream-time (d=192) | 479K | 1.670 | +0.038 |

At matched parameters, the architecture is **+0.085 nats worse** than the transformer. This is a moderate but real efficiency gap — the architecture needs about 2.6× more parameters to reach transformer-comparable quality. This is likely because:
1. The sequential timestep loop prevents parallelism over positions (transformer processes all positions simultaneously)
2. Temporal attention (k=4 window) provides less context than full-sequence self-attention
3. Mix-add bounding may limit representational capacity compared to unbounded residual streams

The gap is not catastrophic (0.085 nats ≈ 5% relative), and the architecture offers properties transformers don't (temporal locality, pipelineable execution, constant memory per timestep). Whether this tradeoff is worthwhile depends on the application. Artifacts: [`experiments/residual_stream_time_partial_detach/artifacts/param_matched/`](../../../experiments/residual_stream_time_partial_detach/artifacts/param_matched/).

### Window size ablation (k=8, k=16): stability constraint

Tested whether larger temporal attention windows improve quality. Result: **both k=8 and k=16 diverge catastrophically** during training.

| Window (k) | Val loss | Outcome |
|------------|----------|---------|
| k=4 (baseline) | 1.670 | Stable (one spike at epoch 4, recovered) |
| k=8 | 7.01 (diverged) | Fatal: train loss hit 573B at epoch 4 |
| k=16 | NaN (diverged) | Fatal: went NaN at epoch 5 |

This reveals a **stability boundary** in the mix-add architecture. The temporal attention mechanism amplifies gradient signals proportionally to window size, and mix-add alone doesn't bound these gradients. The instability seen in the baseline (train spike to 51.02 at epoch 4 that recovered) was a warning — k=4 is already near the edge.

This means k=4 isn't just a convenient choice — it's approximately the largest stable window for this architecture at LR=0.003 with gradient clip 1.0. To increase the temporal context, one would need: lower learning rate, stronger gradient clipping, attention-side normalization (e.g., RMSNorm on Q/K projections), or a fundamentally different temporal coupling mechanism.

**Note:** this does NOT mean the partial-detach sweet spot at N=4 is "because k=4" — the partial detach result was obtained with k=4 fixed. Rather, both results point to the same underlying constraint: the architecture's effective information horizon is ~4 timesteps with current normalization.

Artifacts: [`experiments/residual_stream_time_partial_detach/artifacts/window_ablation/`](../../../experiments/residual_stream_time_partial_detach/artifacts/window_ablation/).

### Window stability fix (LR=0.001)

Tested whether k=8 divergence is a hyperparameter issue by reducing LR from 0.003 to 0.001. Also ran k=4 at LR=0.001 as control.

| Config | Val loss | Stable? |
|--------|----------|---------|
| k=4, LR=0.003 (baseline) | 1.670 | Yes (one spike) |
| **k=8, LR=0.001** | **1.757** | **Yes** |
| k=4, LR=0.001 (control) | 1.781 | Yes |

**k=8 is 0.024 nats BETTER than k=4 at the same LR.** The divergence at LR=0.003 was purely a training stability issue, not architectural. Larger temporal windows DO help — they provide more context for predictions. Both LR=0.001 runs are undertrained (still declining at epoch 5 vs the baseline's 1.670 at the same 5 epochs but higher LR).

Artifacts: [`experiments/residual_stream_time_partial_detach/artifacts/window_stable/`](../../../experiments/residual_stream_time_partial_detach/artifacts/window_stable/).

### Muon optimizer: orthogonal stability hypothesis CONFIRMED

Max's hypothesis ([dictation 2026-05-21-7](../../../dictations/2026-05-21-7.md)): the k=8/k=16 divergence is gradient explosion from repeated weight matrices across time (same W applied T times → Jacobian product W^T). If W drifts away from orthogonal even slightly, that drift compounds to the T-th power. Muon keeps weight matrices near-orthogonal throughout training, so singular values stay ≈1 regardless of T.

**Result: hypothesis fully confirmed.** Muon at LR=0.01 stabilizes all window sizes:

| Window (k) | AdamW LR=0.003 | Muon LR=0.01 |
|------------|---------------|--------------|
| k=4 | 1.670 ✓ | 1.671 ✓ |
| k=8 | **DIVERGED** | **1.664 ✓** |
| k=16 | **DIVERGED** | **1.672 ✓** |

Key observations:
- **k=8 is the best performer** (1.664) — outperforms k=4 at the same training budget
- k=16 is stable but doesn't outperform k=4/k=8 — possibly undertrained (5 epochs) or diminishing returns from the attention window
- Muon default LR=0.02 was too high (all windows diverged); LR=0.01 is the sweet spot for this architecture
- The orthogonality pressure doesn't hurt quality — Muon k=4 (1.671) matches AdamW k=4 (1.670) within noise

This conclusively demonstrates that the "stability boundary" described above was not architectural but optimizer-dependent. The mix-add architecture combined with Muon can train stably at any temporal window size tested. The efficiency gap to transformers (+0.032 at k=8, 479K params) comes from something other than training instability.

Artifacts: [`experiments/muon_window_ablation/artifacts/`](../../../experiments/muon_window_ablation/artifacts/).

### Cosine LR schedule and convergence limits

Flat Muon LR=0.01 diverges at epoch 10 (best val 1.661 at epoch 6). Adding cosine decay (0.01→0.001) prevents divergence but reveals **overfitting on 100k chars** — val loss rises monotonically after epoch 5 while train loss continues falling. Both flat and cosine reach the same best val (~1.661 at 479K, ~1.677 at 184K).

This means the 100k characterization is **data-limited** at this model capacity.

### 900k scale-up: the definitive comparison

To remove the data ceiling, we ran both models head-to-head on 900k/100k TinyShakespeare with matched params:

| Model | Params | Val Loss | Gap |
|-------|--------|----------|-----|
| **Transformer control** | 190K | **1.535** | — |
| Residual-stream-time (Muon, k=8) | 185K | 1.606 | **+0.071** |

Both still improving at epoch 5. Key findings:
- **The gap persists** when data isn't the bottleneck (+0.071 on 900k vs +0.045 on 100k best)
- The transformer benefits MORE from additional data (improved 0.097 from 900k) vs the residual model (improved 0.071)
- This architecture is genuinely less parameter-efficient for text modeling

The efficiency gap is likely intrinsic: k=8 temporal attention window provides much less context mixing than full-sequence attention (ctx=64). The sequential state processing is a fundamental architectural limitation, not a training artifact.

Artifacts: [`experiments/scale_900k/artifacts/`](../../../experiments/scale_900k/artifacts/).

### Partial detach at 900k scale

At 100k, partial detach every N=4 steps cost +0.041 nats but gave 2.46x training speed. At 900k:

| Variant | Val Loss | Time (5 epochs) |
|---------|----------|-----------------|
| Full backprop (reference) | 1.606 | 5650s |
| Partial detach N=4 | **1.606** | 6152s (0.92x — slower) |

**Quality cost: essentially zero** (+0.0008 nats). The model doesn't need long-range gradient flow through time — 4-step backprop is sufficient.

**Speed advantage: gone.** At 900k, the forward pass (temporal attention at each step) dominates wall-clock, not backprop-through-time. The speed benefit seen at 100k was due to BPTT being a larger fraction of total time on a smaller dataset.

Artifacts: [`experiments/partial_detach_900k/artifacts/`](../../../experiments/partial_detach_900k/artifacts/).

### Image patch modality test

Tested on MNIST arbitrary-order patch prediction (32×32, 8×8 grid, 4×4 patches). See [`research/questions/image-patches/`](../image-patches/) for full details.

| Model | Params | MSE | PSNR | Train Time |
|-------|--------|-----|------|-----------|
| Set-transformer (bidirectional) | 229K | **0.050** | **13.0** | **41s** |
| Residual-stream-time (sequential) | 229K | 0.061 | 12.1 | 720s |

**Negative.** The architecture is worse AND 18x slower on image patches. The hypothesized streaming advantage does not materialize — the set-transformer's bidirectional attention is the right inductive bias for this task.

### Orthogonal parameterization (exp-map)

Tested enforced orthogonality (via matrix exponential of skew-symmetric matrices). See [`research/questions/orthogonal-parameterization/`](../orthogonal-parameterization/).

| Model | Params | Val Loss | Train Time |
|-------|--------|----------|-----------|
| Residual-stream + Muon | 185K | 1.606 | ~5 min |
| Residual-stream + exp-map orthogonal | 183K | **1.972** | **108 min** |

**Stability confirmed:** trains stably at k=8 with plain AdamW. This is the third confirmation that orthogonality IS the key stability mechanism.

**Quality destroyed:** +0.366 gap to Muon version. Exact orthogonality over-constrains the network — it can only rotate/reflect, not scale/project. The square-matrix constraint eliminates the FFN expansion bottleneck. Muon's "near-orthogonal" is strictly better than "exactly orthogonal."

---

## Summary of findings

The residual-stream-across-time architecture has been comprehensively characterized:

| Property | Finding |
|----------|---------|
| **Viable?** | Yes — trains, converges, produces reasonable output |
| **Stable?** | With Muon (near-orthogonal optimizer) — yes at any window size |
| **Quality vs transformer (text)?** | Worse: +0.071 at 900k, gap widens with more data |
| **Quality vs transformer (images)?** | Worse: +0.011 MSE, and 18x slower |
| **Speed advantage (partial detach)?** | Zero quality cost but no wall-clock improvement at scale |
| **Orthogonal weights (structural)?** | Stable but over-constrained (+0.366 quality loss) |
| **Key mechanism?** | Orthogonality for stability (confirmed 3 ways) |
| **Practical conclusion?** | Muon is the right approach — optimizer pressure, not structural constraint |

## Open questions this report does not close

- **Can the efficiency gap be closed?** The gap is architectural: k=8 temporal window vs full-sequence attention (ctx=64). Larger windows might help but sequential processing is inherently limited.
- **Does this architecture have advantages in truly streaming/online settings?** Not tested — all experiments used fixed-length batches. A true streaming benchmark might show different results.
- **Could non-square orthogonal-like constraints help?** (e.g., semi-orthogonal: W^T W ≈ I but W not square) — might get stability without the capacity reduction.
- **Is the broader "cortical column" vision dead?** Not necessarily — but the specific instantiation tested (mix-add, temporal attention, sequential processing) doesn't outperform transformers on any metric.

---

*Last updated: 2026-05-22. Full characterization: text, images, orthogonal parameterization, partial detach at scale.*
