# Plan

Working file. Rewrite it as the state changes.

---

## Current state

- Working Python path: UV + `.venv` + CPython 3.12.12 + PyTorch CUDA on RTX 3090.
- **First real positive signal for the predictive chain architecture.** On Shakespeare text (7k chars, context_size=5, 80/20 train/val split), the 8-node detached predictive chain achieves the best validation loss (2.59) of all models tested, despite the highest training loss. It generalizes better than transformer (2.80), RNN (3.05), and feedforward (4.87) baselines.
- The aux weight (0.001 vs 1.0) makes little difference to val performance.
- Detached gradients work — nodes learn independently without harming task performance.
- The tiny repetitive corpus experiments all hit a ceiling (0.9793) and no longer discriminate.

## Key open question

**Why does the predictive chain generalize better?** Hypotheses:
- The message bottleneck between nodes acts as an information bottleneck / regularizer
- The multi-step processing forces more distributed representations
- The auxiliary losses provide implicit regularization pressure
- Some combination of the above

## Possible next experiments (pick one)

1. **Ablate the message bottleneck** — try message_dim = hidden_dim (no bottleneck). If val loss gets worse, the bottleneck is doing the regularization work.
2. **Add attention between neighbors** — Max's vision has nodes attending over neighbor history rather than receiving single messages. Does this help the architecture further?
3. **Increase context size** — go from 5 to 20 on Shakespeare. Does the predictive chain's advantage grow with longer dependencies?
4. **Graph topology** — instead of a line, try a small grid or tree. Does connectivity pattern matter?
5. **Coupled vs detached on Shakespeare** — run the comparison with gradient coupling enabled to see if it matters on the harder task.

Prefer whichever is cheapest and most likely to explain *why* the architecture works.

## Live constraints

- Integrate before starting broad new branches.
- Keep the codebase small.
- Experiment ladder: overfit one batch, tiny end-to-end, inspect outputs, scale.
- Open questions stay open.
