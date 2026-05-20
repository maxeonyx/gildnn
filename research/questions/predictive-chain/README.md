# Predictive Chain — older architecture interpretation

> **⚠ Superseded architecture — historical record**
>
> This experiment was run before the architecture clarification in [dictation 2026-05-20-10](../../../dictations/2026-05-20-10.md). At the time, "node" or "module" in the chain was interpreted as a block with its own hidden dimension and recurrent structure — not a single residual block on a shared `d_model` stream. The dictation clarified that modules should be **single residual blocks on a uniform `d_model` residual stream**. The chain structure tested here does not match that clarified design.
>
> The results are valid evidence about **this specific chain-of-blocks family**. They are not direct evidence about the clarified single-residual-block architecture.

> ⚠️ **Important caveat:** This experiment tests a feedforward pipeline (input → chain → global task head → output). That is **not** Max's actual predictive processing vision. In the real architecture, node A is both input and output — it receives the input and predicts the next input. Deeper nodes have no direct connection to inputs/outputs; they help only through message passing to their neighbors. The global task head used here sidesteps the hard question: can deeper nodes learn to be useful without any direct ground-truth signal? See [dictation 2026-05-20-4](../../../dictations/2026-05-20-4.md). The results below are still informative (local predictive pressure works, unhooked gradients are viable), but the core architectural hypothesis remains untested.

## 1. Origin — which part of the vision this tests

Max's core architectural idea (dictation 3, 2025-05-08-1) is a graph of many small recurrent blocks — cortical columns — each predicting its own next state, communicating via messages, with gradients unhooked at message boundaries so each block learns locally. The latest elaboration (2026-05-19-1) adds: blocks use attention to aggregate from neighbors, and there's a family of small predictive heads — including heads that predict loss — enabling dynamic halting and parallel token sampling at inference time.

The fundamental predictive processing model (dictation 2026-05-20-5): a swath of inputs at time step A, and the immediate internal neighbors predict those inputs at time step B. Deeper nodes are not directly connected to inputs or outputs but still help somehow — that entire space is the hypothesis to explore.

This experiment does not implement that architecture. It implements a deliberate simplification: a **linear chain** of GRU cells with a **global task head that reads all nodes**. The justification for the simplification is: before building the true predictive processing model, it's worth asking whether local predictive pressure and unhooked gradients can work *at all*. That's a prerequisite question. But the simplification also means this experiment doesn't test the hard part — whether deeper nodes can learn useful representations purely through neighbor pressure, without a global readout giving them gradient signal.

What this experiment leaves open, explicitly: what the full graph structure should be, what async execution looks like concretely, whether attention-based aggregation adds value, whether loss-prediction heads are useful for halting, and crucially — whether the architecture works at all without the global task head. All of these remain open questions per Max's own words (2025-05-08-2): *"This is a good question. It's an open question... All of these should be explored."*

---

## 2. Architecture

### The chain

Each node is a `GRUCell`. Node A receives the character embedding. Each subsequent node receives the *message* from the previous node — a low-dimensional projection of the previous node's hidden state. All nodes' final hidden states are concatenated and fed to a task head that predicts the next character.

```
Input embedding
      │
   [Node A]  ──── h_A ──── message_A ──►  [Node B]  ──── h_B ──── message_B ──►  [Node C]  ──── h_C
                                                                                                    │
task head ◄────────────────────────────── concat(h_A, h_B, h_C) ──────────────────────────────────┘
```

Each node also has a **predictive aux head**: A predicts the next token embedding; B predicts A's next message; C predicts B's next message. These auxiliary losses are weighted by `aux_weight` and added to the total loss. With `detach_messages=True`, gradient is stopped at message boundaries — each node cannot backprop into its upstream neighbor.

The 8-node version extends this pattern: A through H in a line, each receiving the previous node's message, each carrying an aux head targeting the upstream neighbor's next output.

```
[A]──msg──►[B]──msg──►[C]──msg──►[D]──msg──►[E]──msg──►[F]──msg──►[G]──msg──►[H]
 ↑pred       ↑pred       ↑pred       ↑pred       ↑pred       ↑pred       ↑pred   ↑pred
(embed)      (A msg)     (B msg)     (C msg)     (D msg)     (E msg)     (F msg) (G msg)
                                       └──────── task head ◄── concat(h_A…h_H) ──┘
```

### Core forward pass (simplified from [`experiments/pytorch_char_predictive_chain.py`](../../../experiments/pytorch_char_predictive_chain.py))

```python
# At each sequence step:
for node_index, (node, message_head) in enumerate(zip(self.nodes, self.message_heads)):
    node_input = (
        embeddings[:, step, :]          # character embedding for node A
        if node_index == 0
        else previous_messages[node_index - 1]  # upstream message for B, C, ...
    )
    hidden = node(node_input, hidden)   # GRUCell update
    message = tanh(message_head(hidden))  # low-dim projection

# Detach messages at boundaries if configured:
previous_messages = [
    msg.detach() if self.detach_messages else msg
    for msg in current_messages[:-1]
]

# After all steps, predict next character from all final hidden states:
logits = task_head(concat(h_A_final, h_B_final, ..., h_N_final))
```

Default dims: embedding 24, hidden 96, message 24 (a bottleneck). The message bottleneck was explicitly ablated — see results.

---

## 3. Hypotheses

**H1: Local predictive pressure can coexist with downstream task learning.**
Adding auxiliary prediction targets to each node won't hurt the task head. The model can pursue both objectives simultaneously.

**H2: Unhooked gradients don't kill the chain.**
With `detach_messages=True`, each node receives no gradient from its downstream neighbors. It learns only from its own aux loss plus indirect signal through the task head (via its hidden state only — the message is detached). The question is whether this is still enough to learn useful representations.

**H3: The chain generalizes better than flat architectures on real text.**
The sequential bottleneck forces information through multiple recurrent stages. Hypothesis: this provides implicit regularization that reduces overfitting on small datasets.

**H4: Adding graph connectivity (skip connections, attention over predecessors) improves on the chain.**
If the linear chain is good, a richer graph should be better — each node can selectively route information from multiple predecessors.

**H5: The chain's advantage scales.**
More data, more parameters — does the chain's structural advantage persist, or is it only a small-data regularization effect?

---

## 4. Results

### H1: Local predictive pressure — confirmed

On a tiny repetitive dataset (overfit regime), task head accuracy hits 1.0 with both `aux_weight=0.001` and `aux_weight=1.0`. Adding strong auxiliary pressure does not harm the task head.

At low weight, aux losses are high (A: 0.51, B: 0.20, C: 0.035) — not because the targets are hard, but because the optimizer wasn't trying. With `aux_weight=1.0`, they drop sharply (A: 0.0086, B: 0.039) while task loss hits 1.27e-07. The optimizer can serve both objectives without conflict.

### H2: Unhooked gradients — confirmed, with a cost

With `detach_messages=True` (gradients stopped at message boundaries), task performance is identical: accuracy 1.0 (overfit), 0.9793 (tiny dataset). Aux losses are moderately higher (overfit total: 0.156 vs 0.085), which makes sense — without downstream gradient pressure, each node only has its own aux loss to structure its messages. It works, but learning is harder.

This is the clearest result for the vision: nodes can learn independently via local prediction alone, without global gradient propagation, and the downstream task head doesn't care.

### H3: Chain generalizes better than flat architectures — confirmed at small scale

Switching to a 7K-char Shakespeare excerpt with an 80/20 train/val split:

| Model | Train Loss | Val Loss | Val Acc |
|-------|-----------|----------|---------|
| Feedforward | 0.79 | 4.87 | 0.282 |
| RNN | 1.23 | 3.05 | 0.292 |
| Transformer | 1.48 | 2.80 | 0.311 |
| Pred Chain (aux 0.001, detached) | 2.04 | **2.59** | 0.281 |
| Pred Chain (aux 1.0, detached) | 1.98 | **2.59** | 0.290 |

The chain has the highest training loss (learns slowly) but the lowest validation loss. The baselines overfit badly — feedforward 6×, RNN and transformer 2–3×. The chain is structurally constrained in a way that prevents this.

Two ablations ruled out the obvious explanations:
- **Aux weight doesn't matter** (both aux settings give val loss 2.59) → generalization advantage is not from aux regularization
- **Message bottleneck doesn't matter** (expanding message_dim from 24 to 96 gives val loss 2.564, slightly *better*) → not from information compression

The advantage is from the multi-hop recurrent structure itself. Information must flow through a strict sequential path; the architecture can't skip steps.

Context size scaling (ctx=5, 10, 20) confirms the advantage is robust:

| Architecture | ctx=5 | ctx=10 | ctx=20 |
|---|---|---|---|
| Predictive chain | **2.59** | **2.57** | **2.63** |
| Transformer | 2.84 | 2.92 | 2.95 |
| RNN | 2.95 | 2.97 | 2.92 |
| Feedforward | 4.71 | 8.81 | 9.75 |

The advantage doesn't monotonically grow with longer dependencies, but it's consistent.

### H4: Graph connectivity improves on chain — refuted

An 8-node DAG where nodes C–H each attend over their immediate predecessor plus one skip connection:

| Model | Val Loss | Val Acc |
|-------|----------|---------|
| Linear chain (13.7K params) | **2.59** | 0.290 |
| Graph uniform_mean (36.4K params) | 2.79 | 0.296 |
| Graph attention (36.4K params) | 2.84 | 0.309 |

Both graph variants are worse than the linear chain — and they have 2.5× more parameters. Adding skip connections gives the model shortcuts around the sequential bottleneck, and it uses them in ways that increase overfitting. The attention mechanism is non-trivial (non-uniform weights at deeper nodes) but doesn't help.

This is the most surprising result. It suggests the chain's advantage *is* the sequential bottleneck, and adding connectivity actively hurts by undoing that constraint.

### H5: Advantage scales — refuted

Scale-up to 100K chars, 200K params, context=32, 5000 steps:

| Architecture | Params | Train Loss | Val Loss | Val Acc |
|---|---|---|---|---|
| RNN | 196K | 1.504 | **1.727** | 0.503 |
| Transformer | 187K | 1.312 | 1.731 | 0.497 |
| Predictive chain | 203K | 1.390 | 1.735 | **0.506** |
| Feedforward | 187K | 0.661 | 3.771 | 0.351 |

> **Standardized-anchor note (added later):** This table is still real, but it is **not** on the repo's current standard comparison frame. The new trust anchors in [`base_experiments/README.md`](../../../base_experiments/README.md) are stronger: transformer `186K` params / best val loss `1.632`, vanilla RNN `186K` params / best val loss `1.706`, both trained for `13` epochs with `AdamW`, `lr=0.003`, `ctx=32`, and reported by **best-of-run** validation loss. The runs in this section used an older setup (`5000` steps, older endpoint reporting, `Adam` rather than `AdamW`, and a different RNN architecture), so the old `1.727` / `1.731` baselines here are undertrained relative to today's anchors. The honest claim is therefore narrower: the predictive chain reached rough parity **against those older baselines**, but it has **not yet been tested against the standardized anchors**. Before treating the chain as a live contender at this scale, it needs a re-run on the standardized frame.

All three recurrent architectures converge to ~1.73 val loss. The chain is no longer best — within noise of the RNN and transformer. With enough data, baselines don't overfit, so the chain's regularization effect becomes irrelevant.

Additional cost: the chain is **10–80× slower** than baselines (1177s vs 15–100s) due to sequential node processing that can't be batched.

---

## 5. What this means for the vision

The preliminary viability question is answered: local predictive learning works, unhooked gradients work, the structural principle is sound. That's a real result.

The regularization finding (H3) is interesting but shouldn't be over-read. It's a small-dataset effect — at scale it disappears. Max's vision is not primarily justified as a regularizer; it's justified by the *practical benefits of asynchronous execution* and the potential interpretability/modularity properties. Those were never tested here.

The graph connectivity result (H4) is worth sitting with. The chain beats the graph at this scale. But the chain is also a much simpler graph — a strictly sequential one. The result may be saying: on a small dataset with simple sequential dependencies (character prediction), learned graph connectivity adds noise more than signal. This doesn't generalize to what Max is actually building — a graph over async computation that can model long-range dependencies across a 2D or higher-dimensional structure. The refutation is real, but it's a refutation of this specific configuration on this specific task, not of the graph idea generally.

The sequential bottleneck interpretation (information must flow through a strict path → implicit regularization) may itself be a confound. The chain also has more recurrent steps than a single-layer RNN, which could independently explain the generalization gap. These weren't separately controlled.

What's *not* settled, per Max's own framing: whether the architecture has qualitative advantages that don't show up in val loss — interpretability, modularity, graceful degradation under node failure. These are potentially more important than the loss numbers.

---

## 6. Next steps — not pursued yet, and why

**Async execution** — the key practical benefit Max described. Not tested because the prerequisite question (does the learning even work?) needed to come first. Now that it does, the natural next experiment would be to implement true async: nodes run on different schedules, communicating via a shared memory structure with staleness. This requires a different execution model, not just a hyperparameter change.

**Larger-scale qualitative study** — interpretability of per-node representations, modularity (can nodes be added/removed/replaced without retraining others?), graceful degradation. Deferred because: (a) the loss advantage disappeared at scale, making it harder to justify large compute, and (b) the chain is already 10–80× slower — running it at meaningful scale for qualitative study is expensive.

**Different tasks (image patches)** — Max explicitly mentioned arbitrary-order image patch prediction (dictation 4). The chain structure has different inductive biases on 2D spatial inputs than on sequential text. Not started — the image pipeline doesn't exist yet.

**Dynamic depth / self-referential loss prediction** — Max described this separately (dictations 2025-05-08-3 and 2026-05-19-1): a prediction head that predicts the loss of the main head, enabling dynamic halting at inference. This was implemented and verified as a standalone experiment in `research/questions/dynamic-depth/`. The natural next combination is dynamic depth *inside* the chain — each node could decide how many recurrent steps to take before passing its message. That combination hasn't been attempted.

**Loss-prediction heads for async triggering** — Max's latest thinking (2026-05-19-1) includes prediction heads that predict the loss of other prediction heads, used to decide *whether to run* a node at the next step. This is the mechanism for async execution. Not modeled here at all.

**Graph attention at scale, on different tasks** — the H4 result (graph connectivity hurts on small Shakespeare) should be revisited with more data and a task where spatial connectivity genuinely matters (e.g., image patches, where skip connections across spatial positions make semantic sense).

---

## Artifacts

- [`artifacts/config.json`](artifacts/config.json) — baseline run config
- [`artifacts/overfit_metrics.json`](artifacts/overfit_metrics.json) — one-batch overfit metrics
- [`artifacts/aux_weight_0.001/`](artifacts/aux_weight_0.001/) — low aux weight run
- [`artifacts/aux_weight_1.0/`](artifacts/aux_weight_1.0/) — strong aux pressure run
- [`artifacts/detached_messages/`](artifacts/detached_messages/) — unhooked gradients run
- [`artifacts/compare_detached_messages.json`](artifacts/compare_detached_messages.json)
- [`artifacts/8_nodes_detached/`](artifacts/8_nodes_detached/) — 8-node chain, detached
- [`artifacts/shakespeare_comparison/`](artifacts/shakespeare_comparison/) — baseline comparison on Shakespeare
- [`artifacts/shakespeare_no_bottleneck/`](artifacts/shakespeare_no_bottleneck/) — message bottleneck ablation
- [`artifacts/shakespeare_graph_attention/`](artifacts/shakespeare_graph_attention/) — DAG with learned attention
- [`artifacts/context_scaling/`](artifacts/context_scaling/) — ctx=5,10,20 sweep
- [`artifacts/scale_up/`](artifacts/scale_up/) — 100K chars, 200K params
