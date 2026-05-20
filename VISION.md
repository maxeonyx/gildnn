# Vision

Personal discovery project. The goal is to understand what these ideas do when actually implemented and run — not to publish, not to compete with anyone. Redoing prior work is fine. Being surprised is good.

## Two independent research threads

These start separately. Combining them is a stretch goal contingent on both making sense individually.

### Thread 1: Cortical column network

Distributed computation across many residual blocks arranged in a graph or schedule, with unusual boundary rules — loosely inspired by the structure of the neocortex, but the biological analogy is motivational, not prescriptive. The interesting questions are engineering and empirical, not biological.

**Architecture clarification (2026-05-20):** a column/node/module is currently understood as a **single residual block** (e.g. one transformer-style block or one FFN block), operating on a **shared residual stream of uniform width `d_model`**. It is not a thick recurrent mini-stack with its own internal hidden dimension. The boundary between modules is simply the residual stream at `d_model`. The interesting design questions are at those boundaries. Recurrence is a possible later direction, not the current thing being tested.

The motivating intuitions:

- A standard transformer processes a sequence synchronously through depth. What if instead you had many residual blocks with nonstandard coupling across depth, time, or graph structure — each governed by different boundary rules?
- Each block predicts its own **next incoming residual stream** (the latent it will receive at the block boundary at the next step) — a local self-supervised objective on the incoming signal, not on the block's own output. Does useful computation emerge from this?
- Predictive heads draw from all residual blocks, including raw inputs — not just the final layer.
- Blocks start as plain residual units. Whether looped reuse across time adds anything useful is a later question, not an assumption.
- Columns communicate through some combination of local graph edges and a global channel. The relative roles of these are not settled.
- The graph should ideally match the GPU architecture — not a theoretical topology imposed on hardware that ignores it.
- Updates propagate based on something like surprisal — a block that hasn't changed much doesn't need to recompute. This could make the system sparse and efficient, or it could be a disaster. Unknown.
- Gradients between blocks are minimally coupled — each block learns somewhat independently, with explicit stop-gradient boundaries. This opens a communication-game dynamic that may or may not lead anywhere useful.

**On looped blocks across time:** if blocks are looped or unrolled across time with attention-style residual connections, the architecture becomes close to a transformer with reused/looped blocks. This is a useful grounding analogy. Stop gradients across time and stop gradients across depth are the same kind of mechanism in this framing.

**Everything above is a hypothesis to be tested, not a design specification.** The open questions below are the actual research agenda.

#### Open questions — Thread 1

These are not rhetorical. They are genuinely unresolved and will each need experiments.

- **What exactly is the block boundary?** If modules are single residual blocks on a uniform `d_model` stream, what information is allowed to cross each boundary unchanged, detached, mixed, or delayed?
- **Which cross-boundary mechanisms matter most?** Stop gradients, attention residual paths, async/shared-memory behavior, selective updates, or some combination?
- **What is the global communication channel?** Current intuition: a stateful broadcast router — a central component that reads from all blocks and broadcasts a single mixed message back. But all-to-all attention is also possible and worth trying. Are there many broadcast channels? Unknown.
- **What exactly is unhooked?** No global gradient propagation is the intent. But embeddings may need partial global gradient. No gradients through time across blocks is likely right but not confirmed. Less gradient coupling = more modularity, but how much less?
- **What is bidirectional propagation?** Bidirectional over graph edges via cross-boundary residual or attention structure? Not designed yet.
- **What triggers a block update?** Currently framed as surprisal — large change in incoming residual stream. But surprise relative to what? Who measures it? What happens during dormancy? Entirely open.
- **What is the I/O contract for the architecture?** How does input enter the system and how does output leave? Entirely open.
- **How real can async execution be?** True per-block event queues are hostile to GPU efficiency. A semi-async masked approximation (masked dense updates, bucketed by activity) is one possible path. Whether "real" async is achievable, or even meaningfully different, is unknown.
- **Does the graph structure matter?** If a global channel does most of the work, graph locality may be cosmetic. This is a key thing to find out — not to assume either way.
- **Do unhooked gradients produce useful specialization or protocol breakdown?** Unknown. Worth finding out.
- **What role, if any, should time-unrolling play?** Since looped blocks across time resemble a transformer with reused blocks, when is that a useful simplification versus a distraction? Open.
- **Whether a block is always full attention+FFN or sometimes FFN-only.** Not settled.

### Thread 2: Dynamic computation depth

A separate idea: use the same model weights multiple times per output token, iterating on internal state, with a learned halting criterion.

The intuition: for easy tokens, one forward pass is enough. For hard tokens, more passes produce better predictions. A loss-prediction head estimates when additional computation is no longer helping, enabling adaptive inference-time compute.

Training: run multiple rollouts per token (1 pass, 2 passes, 4 passes, etc. — exponential schedule). Record loss at each depth. Train the loss-prediction head to predict those losses. At inference, use the head's output to decide when to stop. Dynamic hierarchical encodings might be relevant here — the representation at each depth could encode a different level of abstraction.

This is well-trodden territory. The goal is not novelty — it's understanding whether and how it works on our tasks, and eventually whether it composes with Thread 1.

Starting point: get this working on a standard recurrent model or transformer before involving cortical columns at all.

## GPU utilization as a research question

There is a whole research avenue around understanding what kind of GPU programs fit best on the RTX 3090. Max's hypothesis is that RNNs might get significantly more FLOPs out of a GPU than transformers — but this is genuinely uncertain and needs to be measured. Performance comparisons between architectures should include wall-clock time and actual GPU utilization, not just parameter counts and theoretical FLOPs. The hardware constrains what "fast" means in practice.

## Comparison goals

Before claiming any architecture works, it needs to be compared against:

- **Ordinary transformer baseline** — same parameter count, standard attention + FFN, no recurrence
- **Ordinary RNN** — a basic recurrent model; the simplest possible stateful baseline

These aren't just baselines to beat. They're the reference points that make results interpretable.

## Datasets

**Primary:**

- **Character-level English** — character-by-character language modelling, Karpathy-style. After training, the model should be interactive: type at it, see what it generates.
- **Arbitrary-order image patches** — images split into patches, presented in many orderings during training (raster, reverse raster, vertical, knight's move, random — random should be a high proportion). Model learns to predict any patch given any subset in any order. At inference: fill in missing patches, extend outward, etc. Model should be shape-agnostic over image size. The RNN statefulness matters here: re-presenting patches in a different order requires replaying the RNN from scratch — inference-time task design must account for this.

**Stretch:**

- **Structured prompt-space text** — separate positional encoding spaces for system/developer/user/conversation, rather than a flat sequence. Interesting for self-modifying agent behaviour: appending to the system prompt has different semantics than appending to conversation. Not a priority.
- **Image-text unified** — a natural extension once both modalities work independently.
- Other tasks TBD — see `research/questions/` for emerging candidates.

Note: arbitrary-order text (applying the arbitrary-order patch idea to text tokens) is not a goal — it doesn't make sense as a task in the way arbitrary-order image patches do.

## The Mix-Add operation

A candidate residual update operation from [modularity-loss](https://github.com/maxeonyx/modularity-loss). It aims to be approximately norm-preserving when combining two branch vectors. Max has found it stabilises training in practice, but whether the norm-preservation holds under real conditions is an open question. Worth comparing against plain residual + RMSNorm.

See `research/questions/mix-add/` for the full formula, caveats, and open sub-questions.

## What good outcomes look like

Not "achieved state of the art." Good outcomes here are:

- A working experiment with a clear result — positive or negative — with outputs you can actually look at
- An open question that gets narrower: "we tried X, it did Y, that rules out Z"
- Code that stays small and integrated as complexity grows
- Narratives that are honest about what was found

## Non-goals

- Academic novelty
- Publication
- Beating benchmarks
- A large codebase
- Architectural decisions made by intuition and never tested

## A note on the research object

The most interesting part of the Thread 1 vision is probably not "graph transformer with columns" by itself. It's the combination of modular residual computation, local predictive learning at block boundaries, limited gradient coupling, and nonstandard residual/attention paths — and whether useful distributed computation and emergent communication protocols can arise from that setup.

Recurrence and persistent per-module hidden state are possible later extensions, not current defining traits. The current design question is what happens at block boundaries with a shared `d_model` residual stream.

A risk worth staying alert to: unrestricted all-to-all attention between blocks may bypass the interesting parts entirely. If every block can cheaply read every other's state, the graph becomes decorative and async execution becomes meaningless. Communication channels probably need a bottleneck — or sharply different roles — to preserve the locality the architecture is trying to explore. See `research/questions/broadcast-router/` and `research/questions/graph-vs-global-channel/`.
