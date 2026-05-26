# Question: Does Halting Depth Track Structural Nesting?

## What this asks

When the recurrent depth model with dynamic halting processes a formal grammar with recursive nesting, does it use more iterations for tokens at deeper nesting levels? Specifically: do closing delimiters at nesting depth 4 use more halting depth than closers at depth 1?

This is the strongest test of whether the halt head learns *structural* compute allocation (rather than just character-level frequency effects). Natural language confounds structure with vocabulary — a rare word at shallow nesting still looks "hard." A synthetic grammar isolates nesting depth as the sole source of difficulty.

## Which goals this serves

- **Pathway 5 (Dynamic Depth)** — [ROADMAP.md](../../../ROADMAP.md): "Can a loss predictor decide when recurrent iterations are done?" This tests the complementary question: does the mechanism allocate MORE iterations where structure demands it?
- **Pathway 1 (Wide Recurrent)** — Does recurrent depth develop depth-sensitive representations at all? If iterations are just generic refinement (no structural sensitivity), the recurrent thesis is weaker.
- **Dictation [2026-05-26-3](../../../dictations/2026-05-26-3.md):** "CFG/grammar tasks seem interesting because they're more like programming languages — strict nesting, hierarchical structure."

## The simplification

Tiny model (d=64, 4 heads, ff=256, ctx=64, 8 iterations, ~67K params) on synthetic data with known nesting depth annotations. This is legitimate because:

1. We're testing whether the halt mechanism responds to structural depth *at all* — if the signal exists, it should be visible even at small scale.
2. The grammar is simple enough that the model should learn it well in 5K steps (overfitting is fine — we want perfect predictions on the training distribution so that halting is the only remaining signal).
3. The flat control uses the same characters and vocabulary, isolating nesting as the variable.

If the effect isn't visible at this scale, it's either not there or too weak to matter.

## Grammar design

Three types of binary bracket expressions, each with a unique opener/separator/closer:

```
(E,E)    [E;E]    {E:E}
```

Where `E` is either an atom (`a`, `b`, `c`) or a recursive expression. Max nesting depth: 5. Expressions are newline-separated.

Example at depth 3: `([a;{b:c}],a)`

Each character has a ground-truth nesting depth annotation (openers/separators at their level, closers at the level they close to, atoms at the level of their containing expression).

**Flat control:** Same grammar restricted to max depth 1 — only atoms inside brackets, never nested. Same character vocabulary, same bracket types, same statistical properties except no deep nesting.

## Hypotheses

**H1 (Primary):** Closing delimiters at deeper nesting levels use more halting depth (mean_halt_depth increases monotonically with nesting_depth for closers).

**Rationale:** A closer at depth 4 requires the model to "remember" 4 levels of open brackets and match the correct type. This requires more recurrent processing than a closer at depth 1.

**H2 (Control):** The flat control dataset does NOT show the same depth-sensitivity pattern — closers at depth 0 and depth 1 use similar halting depth.

**Rationale:** If the pattern from H1 appears even without deep nesting, it's a character-level frequency artifact, not structural.

**H3 (Secondary):** Openers also show some depth sensitivity, but weaker than closers.

**Rationale:** An opener at depth 4 must be integrated into a deeper context than one at depth 1, but doesn't require the same "matching" computation.

## Exit conditions

| Outcome | Interpretation | Next step |
|---|---|---|
| H1 confirmed + H2 confirmed | Halting depth genuinely tracks structure | Report as evidence for Pathway 5 structural allocation. Consider scaling test. |
| H1 confirmed but H2 also shows pattern | Character frequency effect, not structural | Investigate what character-level feature drives it. Redesign experiment. |
| H1 not confirmed (flat/random depth across nesting) | Halt head doesn't learn structural allocation | Either the task is too easy, or the mechanism genuinely doesn't capture structure. Try harder grammar or more steps. |

## Planned measurements

### Training metrics (from `runs/grammar_train.py`)

| Metric | Purpose |
|---|---|
| val_loss per 500 steps | Confirm model learns the grammar |
| avg_depth per 500 steps | Track overall depth allocation |
| halt_loss per 500 steps | Confirm halt head is learning |

### Analysis (from `experiments/grammar_depth/analyze_depth.py`)

| Metric | Purpose |
|---|---|
| mean_halt_depth grouped by (token_role × nesting_depth) | Primary H1/H2 evidence |
| count per cell | Verify sufficient samples per condition |

### Measurement coverage check

- Training script logs: val_loss, train_loss, avg_depth, halt_loss, ce_loss ✓
- Analysis script produces: (token_role, nesting_depth, count, mean_halt_depth) table ✓
- Flat control: same scripts, different `--data-dir` ✓

**Gap noted:** Analysis produces mean but not variance/std of halt depth per cell. Mean is sufficient for the primary hypothesis (monotonic increase), but variance would strengthen it (showing consistent allocation, not noise). If results are ambiguous, add std to the analysis script before re-running.

## Configuration

| Parameter | Value | Justification |
|---|---|---|
| d_model | 64 | Smallest size that can hold bracket-matching state |
| n_heads | 4 | Standard ratio |
| ff_dim | 256 | 4× d_model |
| context_size | 64 | Enough for expressions up to depth 5 |
| iterations | 8 | Same as capstone — room for variable allocation |
| steps | 5,000 | Grammar is simple; should converge quickly |
| halt_weight | 0.1 | From capstone — encourages variation without dominating |
| halt_epsilon | 0.0 | No forced minimum depth — let the model choose |
| train chars | 2M | More data than the model can overfit |
| val chars | 100K | Sufficient for reliable statistics |
| max_depth | 5 | Deep enough for clear gradient, shallow enough to be learnable in ctx=64 |

## Commands

```powershell
# Generate data (already done — writes both grammar/ and grammar-flat/ in one run)
.\.venv\Scripts\python.exe -m experiments.grammar_depth.generate_data

# Train (nested grammar — uses default paths: experiments/grammar_depth/artifacts/train/)
.\.venv\Scripts\python.exe -m runs.grammar_train --device cuda

# Analyze (checkpoint saved alongside report)
.\.venv\Scripts\python.exe -m experiments.grammar_depth.analyze_depth --checkpoint experiments/grammar_depth/artifacts/train/checkpoint.pt

# Flat control (override paths to keep artifacts separate)
.\.venv\Scripts\python.exe -m runs.grammar_train --device cuda --data-dir data/grammar-flat --report-path experiments/grammar_depth/artifacts/flat/report.json --log-path experiments/grammar_depth/artifacts/flat/run.jsonl
.\.venv\Scripts\python.exe -m experiments.grammar_depth.analyze_depth --checkpoint experiments/grammar_depth/artifacts/flat/checkpoint.pt
```

## Non-goals

- **Text generation quality** — this is a structural probe, not a generation demo
- **Comparison to transformer baseline** — not needed; we're testing whether the halt head is structurally sensitive, not whether it's better than anything
- **Scale test** — if the effect exists at d=64, scale is interesting but separate; if it doesn't exist at d=64, scaling won't help
- **Learning dynamics** — we only care about the final trained model's depth allocation, not how it gets there

## Results

_Placeholder — to be filled after experiment runs._

## Next steps

_Placeholder — depends on which exit condition is met._
