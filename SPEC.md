# Spec: Local Learning Graph Architecture

> Written 2026-05-31, final day of gildnn project. This is what I would build next — informed by everything learned. Uncertainty is marked. This is a starting point, not a finished design.

---

## The system

A graph of nodes.

Each node:
- Receives inputs from its graph neighbours (stale — delayed by one of the sender's steps)
- Possibly receives raw input (token embeddings, sensory data) as an additional input
- Fires every step (all nodes, synchronously in the simple case)
- Produces outputs that become its neighbours' future inputs
- Achieves its incentives. Internal architecture is irrelevant to this spec.

The graph topology is open — expected to be whatever matches the GPU's memory model best while achieving good task performance. Topology is an experimental variable, not a design constant.

---

## Communication

Lateral connections between nodes carry signals with:

1. **Staleness.** A node sees its neighbour's output from one of the neighbour's steps ago, not the current output. Communication is asynchronous by construction.

2. **Noise bottleneck (principled).** Calibrated noise is injected into the lateral signal. The SNR controls information capacity. This forces compression — a node cannot relay its full internal state; it must select what to communicate. The noise IS the information bottleneck. Noise levels may differ per connection or be learned.

3. **Bidirectional, symmetric mechanism.** Connections exist in both directions with the same mechanism (same projections, same noise). The asymmetry in what actually flows — and how nodes use it — emerges from their different positions in the graph relative to input sources. It is not designed in.

---

## Firing

All nodes fire every step. The graph executes for ~2G steps (where G is graph width) per token, at minimum on the first token of a sequence or on difficult positions. This is slightly wasteful on repetitive input or early timesteps where information hasn't propagated yet, but not a huge deal.

Nodes are sized to extract maximum performance from the hardware. In the ideal version, nodes don't care about synchronization AT ALL — they could be completely different machines running at their own speed. This is why B1 (scalar reward, no shared computation graph) is the target: nodes are fully independent computational units.

Adaptive firing (nodes skip updates when inputs haven't changed) is a future optimization, not a core architectural requirement.

---

## Inputs and outputs

**Inputs spread out.** Raw input can enter at multiple points in the graph. Some nodes receive token embeddings (or projections/subsets) as one of their inputs. These nodes can be anywhere — they don't form a layer or boundary.

**Outputs centralized.** A central head network — NOT a node in the graph — attends over all node states and produces token predictions. The head can be anything (transformer, MLP, whatever). It reads the graph's state and produces logits. The graph's job is to produce states worth reading.

The asymmetry matters: inputs are spread across many nodes, output is one centralized reader of the whole graph. Same structure as: sensory cortex spread across the brain, motor/prefrontal output centralized.

---

## Training: rollouts and multi-horizon prediction

The graph executes rollouts — ~2G steps per input token (where G is graph width). At each rollout step, the central head can make predictions:

- **Variable thinking depth:** Predict next token after 1, 2, 3, 4... steps of graph execution. More steps = more thinking = potentially better predictions.
- **Future token prediction:** At any rollout step, predict not just the next token but tokens +1, +2, +3... ahead.
- **Multi-token output:** Possibly predict multiple tokens from a single graph state (1 step → M tokens).
- **Loss self-prediction:** Heads predict their own loss at each step. This enables dynamic depth (stop thinking when the loss predictor says more steps won't help) and dynamic rollout (know how many future tokens you got right).

At inference: pick the top token at all horizons, or (better) use a small autoregressive predictor fed by the graph state rather than separate CE heads at each horizon. The exact inference strategy is not principled — whatever works.

All training CE losses are aggregated into the broadcast scalar reward that modulates node learning. The nodes don't know about the head directly — they only see the scalar.

This is orthogonal to the local learning question. The multi-horizon training is HOW the network's task performance is measured. The local node objective is HOW nodes learn to produce useful states. They're separate mechanisms connected only by the reward broadcast.

---

## Learning

### The core principle

**Parallelizable computation.** The point of local learning is that nodes can compute and update independently, in parallel. This is both a performance goal (GPU parallelism) and a design constraint (forces the architecture to work without global synchronization).

### One universal objective (THE OPEN QUESTION)

All nodes have the same objective — including nodes connected to raw input. The objective is not differentiated by position in the graph.

**What that objective IS is the central unsolved problem.** This is not a gap in the spec — it IS the research question. Candidates:
- **Predict your next inputs.** Every node predicts what it will receive at its NEXT step from its neighbours (and from raw input, if connected). Temporal prediction, not reconstruction of current inputs.
- **Minimize free energy.** Minimize prediction error about next inputs, subject to a complexity constraint on outputs (noise bottleneck provides this).
- **Something else entirely.** The right answer may not be in the "predict" family at all.

The constraint: whatever it is, it must be universal (same rule for all nodes), local (computable from the node's own inputs/outputs/neighbourhood), and must produce representations that the central head finds useful — even though nodes don't know about the head. The broadcast scalar reward is the only signal connecting node objectives to task performance.

### Why "predict your inputs" failed in this project (and might work with the full system)

The gildnn experiments tested "predict your inputs" with strictly detached laterals. Result: orthogonal to token prediction (gradient cosine 0.013). Features optimized for predicting neighbours are structurally different from features useful for predicting tokens.

The hypothesis for why it COULD work with the full architecture:
- **Noise bottleneck** forces compression (can't just copy — must select what's informative)
- **Broadcast scalar reward** modulates which predictions get reinforced (only reinforce when the central head performed well)
- **Adaptive firing** creates natural temporal abstraction
- **Neighbourhood reward signals** create cooperative pressure

This is the BET. The combination has never been tested. It might not work — in which case the universal objective needs to be something fundamentally different from "predict your inputs."

### Broadcast scalar reward (principled, dopamine-like)

A global scalar — the centralized output's task performance — is broadcast to all nodes, time-delayed. It modulates learning: "the system predicted well this step, reinforce whatever you were doing."

This does not carry gradient or structural information. It's a scalar good/bad signal. Combined with local learning (predicting inputs, Hebbian correlation tracking), it enables credit assignment without backprop.

The time delay is important: the signal arrives AFTER the node has acted, reinforcing/punishing past behaviour. Biologically plausible. Avoids synchronous computation.

### Neighbourhood scalar signal (principled)

Each node's loss value is available to its immediate neighbours, time-delayed by one step. A neighbour can observe: "my output caused my neighbour's prediction error to go up/down." This is a local reward signal — purely between neighbours, no global information, no gradient through the neighbour's computation.

This is variant B1: scalar modulation, independently schedulable, no shared computation graph, biologically plausible.

### Why task signal is NOT missing

The concern "interior nodes don't see task signal" assumes they need gradient from the task. They don't. They need:
1. **The broadcast scalar reward** — tells them WHEN the system did well (temporal credit)
2. **Neighbourhood reward signals** — tells them whether their outputs helped neighbours
3. **Raw input flowing through communication** — task-relevant information IS in the graph, propagating through lateral connections

The task signal is indirect and local. Whether this is SUFFICIENT for useful learning is the open question. But the signal is not absent.

---

## What's NOT in the design

- **Per-node token prediction (per-band CE).** Rejected. Broadcasting the token objective to every node is unprincipled. One centralized output, one task loss.
- **Different objectives for different nodes.** All nodes have the same objective. Position in the graph determines what inputs are available, not what the node is trying to do.
- **Full backprop through the graph.** Explicitly what we're avoiding.
- **Hierarchical layers/bands.** Not a design primitive. Hierarchy may emerge from graph structure. It is not imposed.
- **Fixed rate schedules.** All nodes fire every step. Multi-rate is not part of the core design.

---

## Hacks to avoid unless absolutely necessary

These are scaffolding for early experiments. They should be removable. If the principled design requires them permanently, that's evidence the principled design is wrong.

- **"Predict your neighbour N steps ahead"** — Provides learning signal where neighbourhood reward is too weak. Hacky because N is arbitrary and the objective is hand-designed rather than universal.
- **Fixed multi-rate** — Approximation of temporal abstraction. All nodes fire every step in the real design; multi-rate is a cheap proxy if adaptive skipping is needed later.
- **Fixed/designed topology** — Necessary initially, to be relaxed.
- **Truncated backprop through neighbours (variant A)** — Provides gradient directly, more practical than B1, but NOT biologically plausible (requires computing through neighbour's weights). Use for capability testing. Not the target architecture.
- **Gradient from central head into graph nodes** — If the broadcast scalar alone doesn't ground nodes sufficiently, allowing the head's CE gradient to flow into graph nodes (through the attention) is acceptable as scaffolding. But ideally nodes learn from local objectives + scalar reward only.

---

## The key experimental questions (ordered by importance)

1. **What is the universal node objective?** "Predict your next inputs" is a candidate. There may be others. This is the research question — everything else is engineering.

2. **Does universal objective + noise bottleneck + broadcast reward produce representations the central head can use?** If yes, the architecture works. If no, something fundamental is missing.

3. **What topology works?** Does structure matter, or does any reasonably-connected graph learn? What extracts max performance from the hardware?

4. **How much noise?** What SNR gives the best compression/performance tradeoff?

5. **Is the broadcast scalar reward necessary?** Can neighbourhood signals alone do credit assignment, or is the global scalar required?

6. **How many nodes need raw input?** One? Many? Does density of input connections matter?

---

## Open tensions (real, not resolved)

- **Locality vs learnability.** The project's evidence says strict detachment kills learning. Neighbourhood reward (B1) is less strict — but is it enough? Unknown.
- **Task signal propagation.** Does task-relevance propagate through the graph via communication + reward modulation? Or does it attenuate too fast? Dense graphs (every node within 2-3 hops of raw input) might solve this structurally.
- **Universal objective expressiveness.** "Predict your next inputs" is a specific choice. Is it rich enough to produce diverse useful representations, or does it converge to homogeneous features across nodes?
- **Credit assignment with scalar reward.** REINFORCE-style learning from a global scalar is high variance. Can it learn fast enough, or does credit assignment fail at graph scale?

---

## What this project established

Positive:
- Weight-tied readout works (tied beats untied at matched params)
- Stale laterals have negligible quality cost (+0.005 nats, CI crosses zero)
- CUDA graph execution gives 17x speedup for parallel blocks
- Dynamic depth has huge headroom (learned halting predictor r=0.53)
- Multi-rate gives 20% wall-clock speedup

Negative:
- "Predict neighbour raw state" is orthogonal to token prediction (gradient cosine 0.013) under strict detachment
- Strictly detached laterals cause shallow collapse
- Lateral gradient needed continuously, not just for bootstrapping
- Combined local objectives don't synergize
- No purely-local objective in the "predict neighbours" family produced competitive representations

The gap: neighbourhood reward signals (B1), noise bottleneck, adaptive firing, and broadcast scalar reward were never tested — individually or together. The spec describes an architecture that the project never got to build. It's coherent in theory and unvalidated in practice.
