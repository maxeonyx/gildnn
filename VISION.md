# Vision

Personal discovery project. The goal is to understand what these ideas do when actually implemented and run — not to publish, not to compete with anyone. Redoing prior work is fine. Being surprised is good.

## The core goal: async wall-clock speedup

The central motivation is making recurrent inference *fast*. Not accuracy-first — speed-first. The hypothesis: if you have many small residual blocks that don't need to synchronize with each other, you can keep all parameters resident in GPU memory and process activations in place, replacing depth with width and propagation across timesteps rather than within. Blocks communicate via stale reads from volatile memory. They don't wait for each other. The GPU's many parallel units execute at their own speeds.

This is the thing that might actually give wall-clock speedup. If it doesn't give wall-clock speedup, we're not doing it right yet.

A secondary advantage: modules running at different rates. Some iterate rapidly, some update infrequently, some handle different timescales. Not by hard-coded schedules but by the stale-reads mechanism — pack more into one area of the GPU for a rapidly-iterating module, swap between five in another area, swap between a hundred in a third. Multi-rate execution means a certain part of the network is implicitly attempting to predict further into the future.

**Current experimental status:** Fixed multi-rate gives consistent wall-clock speedup:
- Rates [1, 1, 2, 4] (4 blocks): 14.8% speedup
- Rates [1, 2, 4, 8] (4 blocks): **20.7% speedup** — clears the 20% target
- Rates [1, 2, 4, 8, 16] (5 blocks): ~22% speedup but +0.016 quality cost. Sweet spot is [1,2,4,8].

**Quality nuance (3-seed matched-FLOP result):** Multi-rate beats a same-architecture all-rate-1 baseline per step, but given equal wall-clock compute (wider all-rate-1 model), the wider model is slightly better (+0.018 nats, 3-seed average). Multi-rate's value is the speedup itself — it does less work per step, freeing time. Whether that time advantage nets out positive depends on training budget and whether async execution can further multiply the throughput gain.

## The architecture: diagonal residual connections across time and depth

The design is many separate small residual blocks arranged in a graph. A block is a **single residual block** (e.g. one transformer-style block or one FFN block), operating on a **shared residual stream of uniform width `d_model`**. Not a thick recurrent mini-stack with its own internal hidden dimension. The boundary between modules is simply the residual stream at `d_model`.

The interesting connections are **diagonal** — from residual block A at time t to the next block at time t+1. Residual streams across time, across depth, and diagonally across time and depth. This is the main structural idea to explore.

Each block's job: predict its own next incoming residual stream — a local self-supervised objective on the incoming signal, not on its own output. The first node connected to the input predicts that input at the next step, and that is the output. Deeper nodes learn by producing messages that help their neighbors produce good predictions. Whether deeper nodes learn interesting things given they're not directly connected to input — that's the experiment.

Gradients between blocks are minimally coupled — explicit stop-gradient boundaries. Less gradient coupling = more modularity. How much less is still open, but as unhooked as possible is the direction. Stop gradients across time and stop gradients across depth are the same kind of mechanism in this framing — if you unroll blocks across time with attention-style residual connections, the architecture becomes close to a transformer with reused/looped blocks.

### Self-prediction

The network should predict not just its inputs, but its own outputs and internal latents. Can we compress computation naturally into fewer timesteps? Rather than forcing a network to process one token given its size, have a network that learns to process many tokens or look further ahead. Self-prediction is the training mechanism: a prediction head that estimates what the network itself will produce, trained to match the actual next-step output.

An attention mechanism that predicts attention for the current timestep *and* the next timestep — in training we use the former, in inference we use the latter (which was trained to predict the former at the next step). This allows pipelining computation without waiting for the current step to finish.

### Dynamic depth (computation per token)

Use the same model weights multiple times in sequence to produce a single output token. Iterate on internal state. Train a loss-prediction head that estimates when additional computation is no longer helping. At training: run multiple rollouts (1 pass, 2 passes, 4 passes — exponential schedule), record loss at each depth. The loss predictor learns to predict those losses. At inference: use the head's output to decide when to stop.

This is well-trodden territory. The goal is understanding whether and how it works on our tasks.

### Dynamic token count output

The complementary direction: instead of one token per timestep, predict *many* tokens ahead. A loss-prediction head estimates loss for a given number of output tokens. In training we build that capability; at inference we dynamically select how many tokens to sample without having to separately train for it. The loss predictor over a set of possibilities — for dynamic depth, how many loops; for dynamic token count, how many tokens to sample. Ideally these would be orthogonal.

### The broadcast mechanism

A stateful broadcast component that reads from all modules and broadcasts a mixed message back. Not all-to-all attention between every block — a central bottleneck. It takes in all columns but outputs the same mix to all. Maybe many broadcast channels. It must stale-read from everything (otherwise it forces synchronization, killing throughput) and its outputs must be stale-read by modules.

How the broadcast itself learns what's important is unclear. Maybe self-prediction. Maybe influenced by reward modelling. Without a reward model, probably try without the broadcast — it probably can't learn to do anything useful with purely local learning. This is genuinely open.

**Unrestricted all-to-all attention between blocks may bypass the interesting parts entirely.** If every block cheaply reads every other's state, the graph becomes decorative and async execution meaningless. Communication channels probably need a bottleneck to preserve locality.

### Residual stream management: mix-add over norms

The mix-add operator: `mix(a, b, m) = a * sqrt(σ(m)) + b * sqrt(1 - σ(m))`. A principled way to add two vectors that keeps the distribution approximately normally distributed and norm-preserving. Three variants: fixed hyperparameter, learned static parameter, data-dependent learned parameter. Learned mix-add is preferred — it lets the network set its own forgetting threshold in the residual stream.

Mix-add is not a fix for training stability — it's a fix for having to tune the residual backbone. You can still get exploding activations. Training stability is the optimizer's job.

The preference: attempt architectures without LayerNorm/BatchNorm where possible. Those are hacks unless justified. Worth comparing mix-add against plain residual + norms.

### Muon optimizer for RNN stability

Gradients explode in RNNs because the same weight matrix W applied T times compounds singular values multiplicatively. If W drifts away from orthogonal even slightly, that drift gets amplified to the T-th power. Muon keeps weight matrices near-orthogonal throughout training — orthogonal matrices have singular values exactly 1, so gradients flow back arbitrarily far without exploding or vanishing. Particularly well-suited for architectures where the same module repeats across time.

### Exploratory direction: complex-valued networks and volume-preserving nonlinearities

An interesting side direction (not the main focus): if weight matrices were parameterized to be inherently orthogonal/unitary, activations neither grow nor shrink by construction. Complex-valued networks may offer this. Element-wise nonlinearities privilege a basis (arbitrary, breaks rotational symmetry) — ideally the nonlinearity operates on the geometry of the representation, not individual coordinates. Volume-preserving diffeomorphisms (divergence-free vector fields, Hamiltonian flows) as learned nonlinearities are theoretically attractive. Whether parameterizable cheaply enough is open.

## Open questions

These are genuinely unresolved. Each needs experiments.

- **Does async actually give wall-clock speedup?** The only point of async is speed. If we can't demonstrate it, we're not doing it right yet.
- **What exactly is the block boundary?** What information crosses unchanged, detached, mixed, or delayed?
- **What triggers a block update?** Currently framed as surprisal or fixed rates. But surprise relative to what? Who measures it? Multi-rate by stale reads is one concrete operationalization that works.
- **Do unhooked gradients produce useful specialization or protocol breakdown?** Unknown. Worth finding out.
- **What does the graph structure buy?** If a global channel does most of the work, graph locality may be cosmetic.
- **Can local learning scale?** Does it work at all? Does it work badly? Does it work well but without performance benefits? The exact details of how local modules work probably matter a lot.
- **What GPU programs fit best on the RTX 3090?** RNNs might get significantly more FLOPs out of a GPU than transformers — genuinely uncertain, needs measuring. What's the largest parameter shape that just sits in cache, repeatedly processing data?
- **What role should time-unrolling play?** Since looped blocks across time resemble a transformer with reused blocks, when is that a useful simplification?
- **Compilation backend:** decided PyTorch + `torch.compile` (Triton) for stable paths, custom CUDA/Triton for async research. JAX/IREE possible later but not needed — the hardest part (persistent kernels) goes below both frameworks. See `research/questions/backend-choice/README.md`.

## Datasets

**Primary:**

- **Character-level English** — character-by-character language modelling, Karpathy-style. After training, the model should be interactive: type at it, see what it generates.

**Stretch:**

- **Arbitrary-order image patches** — images split into patches, presented in many orderings during training. Model learns to predict any patch given any subset in any order. Interesting properties: at inference, fill in missing patches, extend outward, choose an optimal sampling order. This connects to Max's master's work (maxeonyx/msc on GitHub). Not the core thing to explore right now — return to it after async is working.
- **Structured prompt-space text** — separate positional encoding spaces for system/developer/user/conversation. Interesting for self-modifying agents. Not a priority.
- **Image-text unified** — a natural extension once both modalities work independently.

Note: arbitrary-order text (applying the arbitrary-order patch idea to text tokens) is not a goal.

## Comparison goals

Before claiming any architecture works, compare against:

- **Ordinary transformer baseline** — same parameter count, standard attention + FFN
- **Ordinary RNN** — the simplest possible stateful baseline (not GRU/LSTM — those are from the past and not what's being explored here; prefer attention over time or mix-add residual across time)

These aren't just baselines to beat. They're the reference points that make results interpretable. Comparisons should include wall-clock time and actual GPU utilization, not just parameter counts and theoretical FLOPs.

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
- Stock RNN mechanisms (GRU, LSTM) unless explicitly compared against the ideas here
