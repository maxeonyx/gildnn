# Local learning on a small RNN

This question serves the local-predictive-learning part of [Max's 2026-05-20 dictation](../../../dictations/2026-05-20-9.md) and the broader Thread 1 goal in [VISION.md](../../../VISION.md): can locally trained recurrent modules, with deliberately limited gradient coupling, do something useful before we bring in async execution, graph scheduling, or richer column machinery?

## Status

First comparison slice partly run. The 1-module and 2-module variants now have full standardized results. The 4-module variant passed the mechanical checks but failed the reduced-setting gate badly enough that it was not promoted to the full comparison table as a fair standardized run.

## The exact question being tested

The first experiment asks a narrower version of the question in the [dictation](../../../dictations/2026-05-20-9.md):

> On the fixed TinyShakespeare comparison frame, does a **multi-module local-learning RNN** beat a **single-module local-learning control** at similar parameter count, and does increasing module count help within the first tested range?

More explicitly, this README is trying to earn an answer to all three of these:

1. Does at least one multi-module local-learning variant beat the same-family single-module local-learning control?
2. Within the tested ladder, does adding modules improve performance, do nothing, or hurt?
3. What wall-clock cost comes with those changes under the same training frame?

The ordinary vanilla RNN baseline from [base_experiments/README.md](../../../base_experiments/README.md) remains a reference point, but it is **not** the primary control for this question. The primary control must stay inside the same local-learning family.

## The simplification being made

This is a legitimate cut from the broader idea, but it is still a cut.

For this first slice:

- we keep the existing char-level TinyShakespeare task,
- we keep synchronous dense execution,
- we keep a simple recurrent backbone,
- we test stop-gradient-separated local learning **without** async or desynchronized execution.

This means the experiment is **not** testing:

- true async execution,
- volatile memory between independently scheduled modules,
- graph topology questions,
- the full cortical-column picture,
- global broadcast/router design,
- surprisal-triggered sparse updates,
- GPU-program-fit claims beyond ordinary wall-clock timing,
- the globally best definition of a "local module".

That last point matters. The dictation says a local module is "probably 3 nodes in the chain," but that is a hypothesis, not a requirement. This README makes one simple first interpretation so the question becomes runnable.

## Initial architecture choice for this first comparison slice

### What a "local module" means here

For this experiment, a **local module** means:

- a small stack of recurrent layers,
- a persistent hidden state internal to that module,
- one local auxiliary prediction head attached to the module,
- a stop-gradient boundary at module interfaces.

The simplest concrete interpretation of "probably 3 nodes" is:

- **one module = 3 recurrent layers** in a small internal stack.

That is a choice for the first slice because it is the smallest interpretation that still gives the module an internal boundary-rich micro-structure instead of collapsing "module" into "single recurrent layer." If this slice is promising, the module depth itself becomes a later question. It is not being swept here.

### What each module predicts

Each module's local head predicts the **next-step input to that same module** rather than the final token target directly. In the shorthand from the dictation, this is the "A -> A" cut: the module learns to model the incoming residual stream it will receive at the next time step.

The final language-model head still predicts the next character from the top-level representation so the whole model remains comparable to the existing text baselines.

### Where stop gradients go

For the first slice, gradients are cut **between modules, not inside a module**.

Concretely:

- within a module's 3-layer recurrent stack, ordinary backprop is allowed,
- each module's local loss updates that module's own parameters and local head,
- the signal passed from module `k` to module `k+1` is detached before module `k+1` uses it,
- the main language-model loss updates only the final readout path and the final module's forward path from its own inputs onward; it does **not** backpropagate through earlier module interfaces.

This is the simplest placement that preserves the intended "minimally coupled" learning story from [VISION.md](../../../VISION.md) without turning the first experiment into a combinatorial sweep over gradient-routing choices.

## Comparison contract

This section defines what "same frame" means for this experiment. If a later run breaks this contract, it does not belong in the same comparison table.

### Fixed task frame

Use the same comparison frame already established in [base_experiments/README.md](../../../base_experiments/README.md):

| Dimension | Contract |
|---|---|
| Dataset | TinyShakespeare (`experiments/corpora.ignore/tinyshakespeare_input.txt`) |
| Train slice | first 100,000 characters |
| Val slice | characters `[100000, 120000)` |
| Objective | character-level next-character prediction |
| Context length | 32 |
| Primary metric | best validation cross-entropy loss (nats/char) |
| Secondary metrics | final val loss, val accuracy, training time |
| Timing metric | wall-clock runtime on this RTX 3090 machine |

### Fixed training-frame controls

Unless a later README revision explicitly says otherwise, all compared variants should keep the same:

- dataset construction,
- train/validation split,
- tokenization/vocabulary,
- batch size,
- epoch budget,
- optimizer family,
- learning-rate schedule,
- gradient clipping rule,
- seed policy,
- logging/reporting format.

The point is to change the local-learning modularity mechanism, not to quietly tune a different training stack per variant.

### Fairness rule for parameter matching

The reference budget is the existing small-text frame at about 186K parameters. For this first slice:

- target budget: **roughly 186K parameters**,
- tolerance: **within ±10%**, matching the existing baseline contract,
- parameter counts must be reported for every variant,
- width should be adjusted as module count changes so that module count is not just acting as a proxy for a larger model.

This tolerance is only tight enough for an initial screen. If results are close enough that the tolerance could change the ranking, the comparison is not yet decisive.

### Fairness rule for timing comparison

Wall-clock comparisons are only meaningful if timing is collected under the same runtime conditions. For this first slice:

- run one training job at a time,
- run on the same machine and same GPU class,
- use the same epoch budget and evaluation cadence,
- report end-to-end training runtime, not just forward-pass microbenchmarks,
- treat wall-clock as real evidence but still machine-specific evidence.

This stage does **not** require a trustworthy FLOPs accounting before runs begin. If FLOPs can be added later without delaying the main answer, great; if not, the wall-clock comparison still stands on its own as the practical hardware-facing measure.

## The control and variant set

This is the minimum variant set intended to answer the question without multiplying dimensions too early.

### Primary control: single-module local-learning control

The primary control is **not** the vanilla RNN baseline.

It is:

- one local-learning module,
- using the same internal module design as the multi-module variants,
- using the same local auxiliary head machinery,
- using the same local target type,
- using the same stop-gradient rule in principle, though with only one module there is no inter-module boundary to detach across.

So the single-module control is the answer to: "what happens if we keep the local-learning machinery but remove modular composition as a degree of freedom?"

That isolates the effect of **multiple modules** from the effect of merely adding auxiliary local-learning machinery.

### Secondary external reference: vanilla RNN baseline

The ordinary RNN anchor from [base_experiments/README.md](../../../base_experiments/README.md) should still be reported beside the local-learning family, because it helps answer a different question:

- is any observed effect specific to local learning,
- or did rebuilding the model family simply make the architecture better or worse for unrelated reasons?

But a win over the vanilla RNN alone would not answer the core question here.

### Initial module-count ladder

Start with the smallest useful ladder:

- **1 module** — single-module local-learning control,
- **2 modules** — smallest true modular comparison,
- **4 modules** — one additional step to see whether "more modules" has any directional signal.

Why not more?

- `1 -> 2` is enough to test "single vs multiple,"
- `2 -> 4` is enough to test whether the first trend continues,
- larger ladders would add cost and interpretation burden before we know this family trains at all.

If `2` is already clearly worse than `1`, the contract expects the matrix to stop there unless there is a concrete reason to think the result is an artifact of the setup rather than the idea.

## Hypotheses being tested

These are the live hypotheses for the first slice:

1. **H0: extra local-learning modularity only hurts.** The single-module local-learning control matches or beats the multi-module variants at similar parameter count.
2. **H1: multiple local-learning modules help.** At least one multi-module variant beats the single-module local-learning control.
3. **H2: there is an early module-count trend.** Moving from 1 to 2 to 4 modules shows either improvement, saturation, or degradation within this slice.
4. **H3: any quality gain may come with a wall-clock tax.** Even if multi-module models help on loss, the gain may or may not justify the extra runtime.

This first slice does **not** distinguish every deeper reason those outcomes might happen. It is a screening experiment, not the final explanation.

## Verification plan summary

Before any full comparison run, the implementation should earn trust in four steps.

### 1. Cheap mechanical checks

For every representative variant:

- parameter count reported,
- forward shapes checked,
- loss-path wiring checked,
- local-head target shapes checked,
- one-batch memorization or equivalent sanity check.

### 2. Explicit gradient-boundary checks

The key non-negotiable check for this experiment is that the chosen stop-gradient boundaries are real.

At minimum, verify all of these:

- a local loss in module `k+1` does **not** produce gradients in module `k` across a detached boundary,
- a module's own local loss does update that module,
- the single-module control still receives meaningful gradients and can optimize,
- the final language-model loss does not silently re-couple earlier modules across intended detach points.

If this cannot be shown directly, the comparison is not yet trustworthy.

### 3. Reduced-setting trainability checks

Before the full frame:

- the single-module local-learning control must learn in a reduced setting,
- at least one multi-module variant must also learn in a reduced setting,
- tiny qualitative samples should look like end-to-end language modelling is functioning rather than numerically diverging.

### 4. Comparison-trust checks

Before making a claim from the full comparison table:

- timing capture must already be part of the run path,
- variance/noise must be low enough, or the result large enough, that the ranking is not obviously just seed luck,
- any very close result should be treated as unresolved until rerun or tightened.

## What this experiment will not settle

Even a clean positive result here would **not** prove:

- that async/desynchronized training works,
- that true columns or graph topology matter,
- that 3-layer modules are the right module boundary,
- that this stop-gradient placement is globally best,
- that local learning beats transformers,
- that RNNs are a better GPU fit than transformers in general,
- that limited-gradient modular learning produces useful emergent communication protocols.

Likewise, a negative result here would only rule against this **tested slice**, not against every possible local-learning design.

## Results

Within this first slice, the answer is currently negative for extra modularity under the tested stop-gradient placement. The 1-module local-learning control reached `1.664525` best validation loss, beating the vanilla RNN anchor's `1.711203`, while the 2-module variant degraded to `1.803579` best validation loss and took substantially longer. The 4-module variant did not earn a full-frame run because it was already weak in the reduced setting.

### Comparison table

| Variant | Params | Best val loss | Final val loss | Val accuracy | Runtime (s) | Notes |
|---|---:|---:|---:|---:|---:|---|
| Single-module local-learning control | 185,489 | 1.664525 | 1.667036 | 0.511118 | 395.11 | Best epoch 9; better than the vanilla RNN anchor on this fixed frame |
| Two-module local-learning | 185,959 | 1.803579 | 2.095361 | 0.484625 | 704.43 | Best epoch 4, then overfit/degraded; worse than both the 1-module control and the vanilla RNN anchor |
| Four-module local-learning | 187,709 | — | — | — | — | Not promoted to the full comparison: tiny-run best val loss only `3.312253` at width `80`, and an auxiliary-weight=`0.0` focused check still failed the memorization gate |
| Vanilla RNN anchor | 186,125 | 1.711203 | 1.733100 | 0.509916 | 160.78 | From [base_experiments/README.md](../../../base_experiments/README.md) |

### Qualitative samples

All full-frame local-learning samples use the same prompt and sample length as the baselines: prompt `First Citizen:\nBefore we proceed`, sample length `320`. Evidence from [`modules_1/sample.txt`](../../../experiments/local_learning/artifacts/modules_1/sample.txt) and [`modules_2/sample.txt`](../../../experiments/local_learning/artifacts/modules_2/sample.txt). The 4-module sample shown here is from the reduced-setting diagnostic, because that variant did not earn a full run.

| Variant | Sample |
|---|---|
| 1 module | ```text
First Citizen:
Before we proceed the people the people the people the people the people the people the people the people the people the people the people the people the people the people the people the people the people the people the people the people the people the people the people the people the people the people the people the people the people 
``` |
| 2 modules | ```text
First Citizen:
Before we proceed them, to the common by the power to the common by the power to the common by the power to the common by the power to the common by the power to the common by the power to the common by the power to the common by the power to the common by the power to the common by the power to the common by the power to the common by
``` |
| 4 modules (tiny diagnostic only) | ```text
First Citizen:
Before we proceed                                                                                                                                                                                                        
``` |

### Gradient-boundary evidence

The non-negotiable check was whether detach boundaries were real rather than assumed. Evidence from [`modules_1/correctness_checks.json`](../../../experiments/local_learning/artifacts/modules_1/correctness_checks.json) and [`modules_2/checks-only/correctness_checks.json`](../../../experiments/local_learning/artifacts/modules_2/checks-only/correctness_checks.json).

Single-module control: the local loss updates the module and local head, while the task loss updates the module and LM head, with no accidental coupling into the local head from the task path.

```json
{
  "single_module_local_loss": {
    "modules": [{"module_index": 1, "grad_norm": 0.16590045412156942, "local_head_grad_norm": 0.09265755392736365}],
    "lm_head_grad_norm": 0.0
  },
  "single_module_task_loss": {
    "modules": [{"module_index": 1, "grad_norm": 1.2483830625997618, "local_head_grad_norm": 0.0}],
    "lm_head_grad_norm": 0.7902555039542101
  }
}
```

Two-module variant: module 2's local loss does not leak gradients back into module 1, and the final language-model loss updates only the top module plus LM head.

```json
{
  "downstream_local_losses": [
    {
      "local_loss_module_index": 2,
      "snapshot": {
        "modules": [
          {"module_index": 1, "grad_norm": 0.0, "local_head_grad_norm": 0.0},
          {"module_index": 2, "grad_norm": 0.08673730416935116, "local_head_grad_norm": 0.047655904975799065}
        ],
        "lm_head_grad_norm": 0.0
      }
    }
  ],
  "task_loss": {
    "modules": [
      {"module_index": 1, "grad_norm": 0.0, "local_head_grad_norm": 0.0},
      {"module_index": 2, "grad_norm": 1.101610490416418, "local_head_grad_norm": 0.0}
    ],
    "lm_head_grad_norm": 0.652943501053914
  }
}
```

### What changed in understanding

What narrowed: for this exact stop-gradient placement and this exact TinyShakespeare frame, adding modules did not help. The 1-module control was already enough to get a real result, and the 2-module variant was both worse and slower. The combination of the 4-module reduced-setting collapse, its width of only `80` hidden units per module at this parameter budget, and the failed focused check with auxiliary weight removed now points more toward an architectural limitation of this detached narrow stack than toward a hidden wiring bug.

What stays open: this does not rule out all local-learning designs, all module boundaries, all local targets, or all credit-routing choices. It only rules against this tested slice: 3-layer local modules, local next-input prediction, and detach-at-every-module-boundary under the ~186K TinyShakespeare frame.

## Next-step decision rule

After the first fair comparison slice:

- if `2` modules clearly beats `1`, continue only with the smallest next run that clarifies whether the gain persists,
- if `2` and `4` both clearly lose to `1`, stop broadening the ladder and either diagnose one concrete failure mode or move on,
- if results are close enough that parameter tolerance or noise could explain them, tighten the contract before making a stronger claim.

That keeps this question small enough to finish inside the project timebox instead of turning into an architecture sprawl.
