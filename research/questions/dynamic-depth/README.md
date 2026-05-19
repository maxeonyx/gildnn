# Dynamic depth on Shakespeare

## Question

Does a weight-shared recurrent model trained with losses at every depth learn a useful loss-prediction signal, and at inference does that signal allocate different amounts of compute to different characters?

## Setup

- Corpus and split follow `experiments/pytorch_char_shakespeare_comparison.py`
- Model: shared `GRUCell`, task head, loss-prediction head
- Training depth: 8
- Comparators: fixed depth 1, fixed depth 8, dynamic depth halting at inference

## Main result

- Fixed depth 1 val loss: 4.3479
- Fixed depth 8 val loss: 4.0664
- Dynamic val loss: 4.3849
- Dynamic mean used depth: 1.354

## Depth allocation

Dynamic inference used this histogram: `{'1': 1042, '2': 274, '3': 58, '4': 16, '5': 7, '6': 6}`

Hardest examples saved in `artifacts/depth_analysis.json`. Annotated text saved in `artifacts/depth_annotation.txt`.

## Artifacts

- Metrics summary: `artifacts/comparison_summary.json`
- Dynamic depth analysis: `artifacts/depth_analysis.json`
- Annotated validation text: `artifacts/depth_annotation.txt`
