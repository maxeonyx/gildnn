# Spec: Local Learning Graph Architecture

> Written 2026-05-31, final day of gildnn project. This is what I would build next — informed by everything learned. Uncertainty is marked. This is a starting point, not a finished design.

---

## The system

A graph of nodes.

Each node:
- Receives inputs from its graph neighbours (stale — delayed by one of the sender's steps)
- Possibly receives raw input (token embeddings, sensory data) as an additional input
- Fires when triggered (accumulation/information-based, not a fixed schedule)
- Produces outputs that become its neighbours' future inputs
- Achieves its incentives. Internal architecture is irrelevant to this spec.

The graph topology is open — expected to be whatever matches the GPU's memory model best while achieving good task performance. Topology is an experimental variable, not a design constant.

---

## Communication

Lateral connections between nodes carry signals with:

1. **Staleness.** A node sees its neighbour's output from one of the neighbour's steps ago, not the current output. Communication is asynchronous by construction.

2. **Noise bottleneck (principled).** Calibrated noise is injected into the lateral signal. The SNR controls information capacity. This forces compression — a node cannot relay its full internal state; it must select what to communicate. The noise IS the information bottleneck. Noise levels may differ per connection or be learned.

3. **Bidirectional but not symmetric.** Connections exist in both directions. The two directions have different roles: "toward raw input" and "away from raw input." These are not hierarchical layers — just relative position in the graph w.r.t. where raw input enters.

---

## Firing trigger

Nodes fire based on an accumulation or information-based trigger — NOT a fixed schedule. The principle: temporal abstraction emerges from when nodes choose to fire, not from a designer-imposed rate. Nodes near raw input (which changes every token) fire frequently. Interior nodes that see slowly-changing signals fire less often.

The exact trigger mechanism is an open experimental question (prediction error threshold, information accumulation, learned gate, etc.).

---

## Inputs and outputs

**Inputs spread out.** Raw input can enter at multiple points in the graph. Some nodes receive token embeddings (or projections/subsets) as one of their inputs. These nodes can be anywhere — they don't form a layer or boundary.

**Outputs centralized.** For the task, there is a single readout point. For language: weight-tied normalized readout producing token probabilities. For mixed modality (vision → motor, etc.): the output node(s) produce a control signal or latent.

The asymmetry matters: many nodes are grounded by raw input, but the system's task performance is measured at one centralized output. This is the same structure as: sensory cortex spread across the brain, motor output centralized.

---

## Learning

### The core principle

**Parallelizable computation.** The point of local learning is that nodes can compute and update independently, in parallel. This is both a performance goal (GPU parallelism) and a design constraint (forces the architecture to work without global synchronization).

### One universal objective

All nodes have the same objective — including nodes connected to raw input. The objective is not differentiated by position in the graph.

What that objective IS remains the key open question. Candidates:
- **Predict your inputs.** Every node tries to predict what it will receive next from its neighbours (and from raw input, if connected). Input-connected nodes are naturally grounded by reality because raw input is part of what they predict. Interior nodes predict their neighbours' outputs.
- **Minimize surprise (free energy).** Same as above, framed as: minimize prediction error about inputs, subject to a complexity constraint on outputs (provided by the noise bottleneck).
- **Some other formulation** that achieves the same properties: local, universal, grounded-by-connection-to-reality.

The key insight: you don't need a DIFFERENT objective for input-connected nodes. The universal objective + the fact that some inputs are raw data = task grounding without special-casing.

### Why "predict your inputs" alone failed in this project

The gildnn experiments tested "predict your inputs" with strictly detached laterals and found it orthogonal to token prediction (gradient cosine 0.013). The features optimized for predicting neighbours are structurally different from features useful for predicting tokens.

The hypothesis for why it COULD work with the full architecture:
- **Noise bottleneck** forces compression (can't just copy neighbour states — must select what's informative)
- **Broadcast scalar reward** modulates which predictions get reinforced (only reinforce when the system's centralized output performed well)
- **Adaptive firing** creates natural temporal abstraction
- **Neighbourhood interactions** create cooperative pressure

This is the BET — not established truth. The combination has never been tested.

### Broadcast scalar reward (principled, dopamine-like)

A global scalar — the centralized output's task performance — is broadcast to all nodes, time-delayed. It modulates learning: "the system predicted well this step, reinforce whatever you were doing."

This does not carry gradient or structural information. It's a scalar good/bad signal. Combined with local learning (predicting inputs, Hebbian correlation tracking), it enables credit assignment without backprop.

The time delay is important: the signal arrives AFTER the node has acted, reinforcing/punishing past behaviour. Biologically plausible. Avoids synchronous computation.

### Neighbourhood scalar signal (principled)

Each node's loss value is available to its immediate neighbours, time-delayed by one step. A neighbour can observe: "my output caused my neighbour's prediction error to go up/down." This is a local reward signal — purely between neighbours, no global information, no gradient through the neighbour's computation.

This is variant B1: scalar modulation, independently schedulable, no shared computation graph, biologically plausible.

### Why task signal is NOT missing

The concern "interior nodes don't see task signal" is wrong. The graph is connected. Raw input enters at some nodes. Those nodes' outputs flow to neighbours. Neighbours' outputs flow to their neighbours. The task-relevant information IS in the graph — it propagates through communication, not through gradient.

The broadcast scalar reward provides the WHICH (which timesteps were good), and the local dynamics provide the WHAT (what to do differently). Together: credit assignment without backprop.

Whether this is SUFFICIENT for learning useful representations is the open experimental question. But the signal is not absent — it's indirect and local.

---

## What's NOT in the design

- **Per-node token prediction (per-band CE).** Rejected. Broadcasting the token objective to every node is unprincipled. One centralized output, one task loss.
- **Different objectives for different nodes.** All nodes have the same objective. Position in the graph determines what inputs are available, not what the node is trying to do.
- **Full backprop through the graph.** Explicitly what we're avoiding.
- **Hierarchical layers/bands.** Not a design primitive. Hierarchy may emerge from graph distance to raw input and from firing patterns. It is not imposed.
- **Fixed rate schedules.** Hack for early experiments. Real system uses information-based triggers.

---

## Hacks to avoid unless absolutely necessary

These are scaffolding for early experiments. They should be removable. If the principled design requires them permanently, that's evidence the principled design is wrong.

- **"Predict your neighbour N steps ahead"** — Provides learning signal where neighbourhood reward is too weak. Hacky because N is arbitrary and the objective is hand-designed rather than universal.
- **Fixed multi-rate** — Approximation of adaptive firing.
- **Fixed/designed topology** — Necessary initially, to be relaxed.
- **Truncated backprop through neighbours (variant A)** — Provides gradient directly, more practical than B1, but NOT biologically plausible (requires computing through neighbour's weights). Use for capability testing. Not the target architecture.
- **Separate readout loss feeding gradient into input-connected nodes** — If the universal objective alone doesn't ground input-connected nodes sufficiently, a CE loss on the readout is acceptable as scaffolding. But ideally the universal objective + raw input connection is enough.

---

## The key experimental questions (ordered by importance)

1. **Does the universal objective + noise bottleneck + broadcast reward produce useful representations?** This is everything. If yes, the architecture works. If no, something fundamental is missing.

2. **What is the right firing trigger?** Does temporal abstraction actually emerge from adaptive firing?

3. **What topology works?** Does structure matter, or does any reasonably-connected graph learn?

4. **How much noise?** What SNR gives the best compression/performance tradeoff?

5. **Is the broadcast scalar reward necessary?** Can neighbourhood signals alone do credit assignment, or is the global scalar required?

6. **How many nodes need raw input?** One? Many? All boundary nodes? Does density of input connections matter?

---

## Open tensions (real, not resolved)

- **Locality vs learnability.** The project's evidence says strict detachment kills learning. Neighbourhood reward (B1) is less strict — but is it enough? Unknown.
- **Task signal propagation.** Does task-relevance propagate through the graph via communication + reward modulation? Or does it attenuate too fast? Dense graphs (every node within 2-3 hops of raw input) might solve this structurally.
- **Adaptive firing stability.** Nodes that fire rarely get less learning signal. Does the broadcast scalar compensate? Or do slow nodes become spectators?
- **Universal objective expressiveness.** "Predict your inputs" is a specific choice. Is it rich enough to produce diverse useful representations, or does it converge to homogeneous features across nodes?

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
