# MNIST Baseline Experiment

This notebook captures the latest MNIST baseline run driven by the shared gildnn experiment infrastructure.

## Overview

<!-- SECTION:overview start -->
<!-- Explain how this baseline contributes to the evolving core implementation. -->
<!-- SECTION:overview end -->

## Hypotheses

<!-- SECTION:hypotheses start -->
<!-- Document the expectations we plan to validate before training. -->
<!-- SECTION:hypotheses end -->

## Configuration

<!-- SECTION:configuration start -->
- Seed: 1337
- Batch size: 64
- Hidden units: 128
- Train steps: 200
- Learning rate: 0.0010
- Five-shot evaluation batches: 5
<!-- SECTION:configuration end -->

## Metrics

<!-- SECTION:metrics start -->
- Step 10 five-shot loss: 1.9190
- Step 10 five-shot accuracy: 64.38%
- Final train loss: 0.4411
- Final train accuracy: 85.94%
- Final test loss: 0.3751
- Final test accuracy: 89.29%

| Step | Train Loss | Train Accuracy (%) |
| --- | --- | --- |
| 1 | 2.2989 | 17.19 |
| 5 | 2.1333 | 56.25 |
| 10 | 2.0106 | 60.94 |
| 25 | 1.2709 | 76.56 |
| 50 | 0.6762 | 92.19 |
| 100 | 0.3057 | 92.19 |
| 150 | 0.4862 | 89.06 |
| 200 | 0.4411 | 85.94 |
<!-- SECTION:metrics end -->

## Sample Artifacts

<!-- SECTION:samples-primary start -->
### Training split

#### Sample 1 (index 0)
- True label: 5
- Predicted: 3

![Sample image](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAABe0lEQVR4Ae3AA6AkWZbG8f937o3IzKdyS2Oubdu2bdu2bdu2bWmMnpZKr54yMyLu+Xa3anqmhztr1a/y/AFQef4AqDx/AFSePwAqzx8AwQOUkydPfuYX/9QNP+jlZwFUnumW/lVf/fjbAdzxtW+z/ze/AyCueJnfOMYV+b6H3HXxiQDiipN/8lCAP9l9neEYAACFK5a3H33Xm/DXr/vdP731cwAAiPvt7H/L+73HD/JMAAT32/Ml3j94JgAKz/aHr/paf/RUrgBAPMDD/nL3t/78GwwAQOEBLj7xnV7hTVZPOwAAKDzQE37hEQ99vc2/3gcAxHM6/hbfpd98AwBAPLd1nd7otwGg8hxe8u1fofK43wUAKg/wqI94m+ug3Z0AQOVZrnvXD3sw8Odf8LMAAJVnuvbFvu7RwJ982c8kAAAVAE5+y0s/FPjDr/iVJZcBUAFe6RNe8UZg+TVfeMgzAVAB3uZt4PE/1758l2cBQDx/AFSePwAqzx8A/wiTjVHnBWqmBAAAAABJRU5ErkJggg==)

#### Sample 2 (index 1)
- True label: 0
- Predicted: 0

![Sample image](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAABhElEQVR4Ae3AA6AkWZbG8f937o3IzKdyS2Oubdu2bdu2bdu2bWmMnpZKr54yMyLu+Xa3anqmhztr1a/y/AFQef4AqDx/AFSePwCC5/Ry39W+62UBAMRzeOnf3IFLpwCA4IFe8eeOsX/+2Kv0ABA828ar/+j18OQP4/c/HgCCZ/uW374JeNmt3+ElACB4lpd7M+l3P0H3/NXXhgBA3O+lf3OHX3qX13rJbz9LO3qtvwQInumRn3Ds/N9/z8EvfNFZYPFxAFSumH35m+6/558veKZbAKhc8bJvylv9Ds8GQHDFV+h3f4f7hQRABYA3f2n/LM+S/msAAgAW/X0/wjPNvojf/GQAKs+0vpsrZp/+CXd8xQEAlWf6Wa546U94p595OwAgAEB6ay772N985x98OwCACgD2dV/7nedf+T1e6qbbfuUbAQCoPFP50LfbewT80W9+JgAACABu+rFXQOb8D38UVwBQAGDvF49eU3zNh/4AzwSAeP4AqDx/AFSePwAqzx8A/wgfB01mj9wiKwAAAABJRU5ErkJggg==)

#### Sample 3 (index 2)
- True label: 4
- Predicted: 4

![Sample image](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAABK0lEQVR4Ae3AA6AkWZbG8f937o3IzKdyS2Oubdu2bdu2bdu2bWmMnpZKr54yMyLu+Xa3anqmhztr1a/y/AFQef4AqDx/AFSePwAqzx8AwfPxevc8CoAAeM234Tm8wp8DQAC89tvyQPGQWwRAALwnz+H6D/jrJwBQAYLn8O08GQAq8JLX8hyO8WsAUIE3XfBA1z6EOwGgAo/iH3iAL7/2SfsAUAH4M+6388bv/oZ83i4AVABOAi8Vr3dT/26x/JN1/QsAQMA3ftDubfCSmo4e9yd//jv33nGiBwAq8KHPeFXgtp953B8DfOCZpwEAVIAv4QFej58AAKg8r58GAKg8fwAEz0OPAACoPA8HAEDwvF4FAKDyPAQAQPDcfskAAIjnD4DK8wdA5fkD4B8BdVsrD11rA58AAAAASUVORK5CYII=)
<!-- SECTION:samples-primary end -->

<!-- SECTION:samples-secondary start -->
### Test split

#### Sample 1 (index 0)
- True label: 7
- Predicted: 7

![Sample image](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAABR0lEQVR4Ae3AA6AkWZbG8f937o3IzKdyS2Oubdu2bdu2bdu2bWmMnpZKr54yMyLu+Xa3anqmhztr1a/y/AFQef4AqDx/AFSePwAqzx8AlecPgMrzB0AFgLf/gLtWP3DPU3gWAAQAT3swsP8PXHHHl/45ABUAPuClHvfYl3ntV779ZpjOXs9tfw6AeLYTL/PnrwCrJz3+5Id/IwDiebzdj/7961wAQDy3a/7umrf/CQCoPLcPO3PxiQCAeC6v9pvda/8uAFB5Lm/a/cYfAQBUntPijYfPGgEAguf0CS/zm38IAIB4Dm/204dv8kcAAFQe6NTXll/8IwAAEA9Q/vjlnvrGTwUAAPEAj3wCb/VzXAZA5dke9Kt8ws9zBQCVZ/vAW/gdcwUAlWd5jY/g2QCoPMurb/HUA54JgMoD/M3rXeCZABDPHwD/CNQiOXn1q2tvAAAAAElFTkSuQmCC)

#### Sample 2 (index 1)
- True label: 2
- Predicted: 2

![Sample image](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAABZUlEQVR4Ae3AA6AkWZbG8f937o3IzKdyS2Oubdu2bdu2bdu2bWmMnpZKr54yMyLu+Xa3anqmhztr1a/y/AFQef4AqDx/AFSe5eMXL/n2fNMffR8AAIj7/cjbA/DU178NAIDCM/3I28MTfuDso06e/z0AACpXvPzb8A9vee6g/+OXOgkAAJUrrtc/vNHd8PGP5RcAAKByxc89fP8C8E4dVwBQeaZnAHzCI/mTPwEAAPEAb/5j/X3v/DsAAFB5gJfv+ZHf4TIAKs/202/I9346VwAgnuX6vzl17lWfyhUAVJ7lJ07x/U/lmQCo3O8tX5bf/izuB0DlmU59asdfH3A/ACrP9HGvwE9/Fs8CgHimVcdNd/MsAFSe7eQIXBq7Y5z4GGifdFR5tr8F+LG7r30nALjnC8Qz/eRb8SxT8rN/zu//kbjfJ3bwYu8E33krP/l4ABDPHwCV5w+AyvMHQOX5A6Dy/AHwj6mYQe/2BEjqAAAAAElFTkSuQmCC)

#### Sample 3 (index 2)
- True label: 1
- Predicted: 1

![Sample image](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAABFUlEQVR4Ae3AA6AkWZbG8f937o3IzKdyS2Oubdu2bdu2bdu2bWmMnpZKr54yMyLu+Xa3anqmhztr1a/y/AFQef4AqDx/AFSePwAKz+WRZy/8KQBA5bm8TN4JAEDlubz04U8CABA8p5f4iB8AAIDKc3rUxo8AAIB4Tn965sUPAQAInsODX359CAAAlefwWpzlMgAqz+El+FIuA0A80Kv8/K2vtgIAgOCBXu/kE1ZcBkDlgV7KP84VAIgHuO6vLz6GKwAIHuC9r/ljngmAygM8iIs8EwCVB3gLfp5nAqDybK9xLc8CQOXZ3rr81e/wTABUnmXjTfnxxjMBIJ6l+5373vWIZwJAPH8AVJ4/ACrPHwCV5w+AfwTdxyLFzvAGEgAAAABJRU5ErkJggg==)
<!-- SECTION:samples-secondary end -->
