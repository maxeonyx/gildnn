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

The following branches cover the space of meaningful choices. Not all combinations are sensible — pruning reasoning follows each branch.

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

**C. Causal attention over past residual states** *(Max's preferred mechanism)*
The block at time T has access to a short history `[s_{T-k}, ..., s_{T-1}]` and computes attention-weighted reads over those states, then adds to the stream:
```
context = CausalAttn([s_{T-k}, ..., s_{T-1}])
s_T = s_{T-1} + block(s_{T-1}, x_T, context)
```
This is what Max explicitly asks for in [2026-05-20-15](../../../dictations/2026-05-20-15.md): *"I'm interested in attention over the past."* The window length `k` is a design choice — short window is cheap and testable; long window approaches full sequence attention over the temporal dimension.

**D. Attention + mix-add**
Combine C and B: attention over past states, with mix-add combining the result into the stream. This is probably not the first thing to test — it conflates two variables.

**Pruned: gating mechanisms (GRU, LSTM)**
Max is explicit: *"I don't like these gating mechanisms, I like residual streams."* The GRU reset/update gate is a forgetting mechanism. It is not compatible with the accumulation primitive Max wants to explore. These are excluded from the search space regardless of their empirical performance.

### 2. Residual topology

How do blocks connect across time?

**A. Same block across time (looped)**
One block `f` is applied at every timestep. The same weights process the stream at T=0, T=1, T=2, ... This is weight-tied recurrence, but without gating. It directly implements the grounding analogy Max gave: *"a transformer with attention residual connections, but with looped blocks."*

**B. Diagonal block coupling**
From [2026-05-20-15](../../../dictations/2026-05-20-15.md): *"Going from residual block A to the next block in the next time step, a diagonal step."* In a multi-block arrangement (say blocks B1, B2, B3 running in sequence within each timestep), instead of B1 at time T only seeing B3's output at T (pure depth coupling), B1 at time T also sees B1's output at T-1 via a residual add. Block Bi at time T reads from block B_{i-1} at T and from block Bi at T-1. This is a diagonal in the (depth × time) grid.

Visually:
```
       T=0    T=1    T=2
B3:    [  ]-->[  ]-->[  ]
        ↑      ↑      ↑
B2:    [  ]-->[  ]-->[  ]
        ↑      ↑      ↑
B1:    [  ]-->[  ]-->[  ]
```
In pure depth, arrows are only vertical (↑). In diagonal coupling, each block also has a horizontal arrow from its own previous timestep. The diagonal residual is the horizontal arrow.

**C. Both**
The full topology: all blocks reused across time, with diagonal residual connections in addition to the standard depth-wise residuals. This is the richest version but also the hardest to build cleanly as a first probe.

**Minimal faithful choice:** A is the first thing to test. B requires a multi-block setup; it is a meaningful experiment but should come after A works.

### 3. Gradient coupling

Whether and where backpropagation flows across timestep boundaries.

**A. Fully end-to-end**
No stop-gradients. Gradients flow through the temporal residual connection across all timesteps. This is the baseline: does the architecture learn at all?

**B. Stop-gradient at every timestep boundary**
`s_{T-1}` is treated as a constant when computing gradients for time T. Each block only updates based on local loss at T. This is what enables async execution to be meaningful for training: if block B1's gradient at T=5 doesn't depend on B2's activations at T=4, B2 doesn't have to finish T=4 before B1 can start T=5's backward pass. Max explicitly connects stop-gradients to async performance in [2026-05-20-15](../../../dictations/2026-05-20-15.md): *"I also do expect stop gradients to be important for async performance in training."*

**C. Stop-gradient at selected boundaries only**
Partial decoupling: detach across certain time boundaries, keep full gradients elsewhere. This is probably not the first thing to explore — it adds a hyperparameter without a clear principled choice for where to put the boundaries.

**What is known:** stop-gradient at timestep boundaries is likely needed to realize the async execution speedup. Whether it also helps or hurts training *quality* is unknown and should be measured. These are two separate questions.

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

The smallest model that honestly instantiates this architecture, as Max described it.

### What it is

A character-level language model on the Karpathy dataset with the following structure:

- **One residual block**, `B`, with standard transformer-style internals (LayerNorm → Attention or FFN → residual add). No separate hidden dimension; block input and output are both `d_model`.
- **Unrolled across time** for a sequence of `T` input characters. At each timestep T, the block takes `s_{T-1}` (the residual stream from the prior timestep) and the input embedding `x_T`, and produces:
  ```
  s_T = s_{T-1} + B(s_{T-1}, x_T)
  ```
- **Optional: causal attention over a short history.** The block can attend over `[s_{T-k}, ..., s_{T-1}]` when computing its update. Window size `k` is a hyperparameter (start with `k=4` or `k=8`). This tests mechanism 1C vs 1A in the same codebase by setting `k=0` for the ablation.
- **Output head:** linear projection from `s_T` to vocabulary logits. Cross-entropy loss against next character.
- **No broadcast. No local learning. No async execution.** These are orthogonal variables to isolate later.
- **Stop-gradient variant:** run with and without `detach(s_{T-1})` at the temporal boundary. Measure whether this degrades loss and by how much (this quantifies the cost of the async-enabling change).

### Why this is faithful

- Uses residual addition across time, not gating — correct primitive.
- Reuses one block across time — directly realizes Max's "looped blocks" description.
- Causal attention over past residual states — Max's preferred temporal coupling mechanism.
- No GRU, no LSTM, no built-in recurrent unit.
- The stop-gradient variant measures the async-enabling mechanism in isolation.

### What this probe can tell us

- Can this architecture learn to model character-level language at all? (sanity check)
- Does causal attention over past residual states improve on plain residual add? (mechanism 1C vs 1A)
- What does stop-gradient at timestep boundaries cost in loss? (quantifies the async-enabling tradeoff)
- Is mix-add better than plain add at the temporal boundary? (comparison arm)

### What this probe cannot tell us

- Whether diagonal block coupling (mechanism 2B) helps — needs multi-block architecture.
- Whether local learning works in this frame — deferred.
- Whether the broadcast mechanism can learn anything useful — deferred.
- Whether true async execution gives throughput gains — deferred; requires systems work after the architecture is validated.
- Whether this scales — one tiny probe on char-level language is evidence of tractability, not performance.

### Comparison against wrong baseline

The probe should include a comparison against a standard GRU on the same task and same parameter count. Not because GRU is the goal — it isn't — but because it is the thing that has been built and run before, and we need to show that the new architecture is at least in the same ballpark before scaling it. If the plain residual-stream-across-time model dramatically underperforms a GRU on a trivial task, that is important information about whether the primitive is even viable, not evidence that we should switch back to GRUs.

---

## Open questions this report does not close

These remain genuinely open after this analysis:

- **Does the temporal residual stream actually learn anything useful in practice?** Unknown until the probe runs.
- **What window size for causal attention over past states?** Not derivable from first principles; needs ablation.
- **Where exactly do stop-gradients go?** Per-timestep boundary is one choice; per-block-boundary-within-timestep is another. Exact placement matters for what "local" means.
- **Can local learning work without end-to-end gradients?** The concept analysis above identifies the key failure modes but doesn't resolve them.
- **Does the broadcast require a reward signal?** Max is tentatively yes but explicitly uncertain.
- **What is the right d_model for a first probe?** Small enough to iterate fast; large enough for attention to do something. Starting guess: 128 or 256.

---

*Last updated: 2026-05-21. Theory analysis only — no experimental results yet.*
