# Dynamic depth feature analysis

## Reproduction

- Threshold: 1.39723
- Validation tokens analyzed: 19968
- Reproduced mean depth: 4.7265
- Reproduced validation loss at used depth: 1.731691
- Reference mean depth from saved sweep: 4.5619
- Reference validation loss from saved sweep: 1.738072

## Effect sizes

| Rank | Feature | Metric | Abs effect size | Signed effect | Notes |
| --- | --- | --- | ---: | ---: | --- |
| 1 | position_in_word | correlation_ratio_eta | 0.4611 | +0.4611 | word starts/ends vs middle/non-word |
| 2 | character_identity | correlation_ratio_eta | 0.3745 | +0.3745 | depth spread across target characters |
| 3 | after_punctuation | point_biserial_r | 0.2312 | -0.2312 | immediate post-punctuation positions are shallower here |
| 4 | bigram_novelty | spearman_rho | 0.1899 | -0.1899 | higher novelty is shallower in this reproduction |
| 5 | word_frequency | spearman_rho | 0.1607 | +0.1607 | rarer words get slightly deeper compute |
| 6 | local_entropy | spearman_rho | 0.1171 | -0.1171 | nearby next-character entropy |

## Character identity

- Effect size (`summary.json.character_identity.effect_size`): 0.3745
- Highest-mean-depth characters (`summary.json.character_identity.top_characters`):

| Character | Mean depth | Count |
| --- | ---: | ---: |
| B | 8.0000 | 30 |
| V | 8.0000 | 22 |
| Y | 8.0000 | 16 |
| q | 8.0000 | 6 |
| j | 8.0000 | 5 |
| J | 8.0000 | 3 |
| K | 8.0000 | 1 |
| b | 7.8912 | 193 |
| W | 7.8333 | 66 |
| H | 7.5357 | 28 |

- Lowest-mean-depth characters (`summary.json.character_identity.bottom_characters`):

| Character | Mean depth | Count |
| --- | ---: | ---: |
| E | 1.3103 | 29 |
| U | 1.3592 | 103 |
| \n | 2.1304 | 805 |
| N | 2.1667 | 90 |
| z | 2.7500 | 8 |
| L | 2.7949 | 39 |
| O | 2.8919 | 74 |
| <space> | 3.3121 | 2903 |
| u | 3.9461 | 445 |
| h | 3.9989 | 907 |

## Bigram novelty

- Spearman rho (`summary.json.bigram_novelty.effect_size`): -0.1899
- Mean depth by novelty quintile (`summary.json.bigram_novelty.mean_depth_by_bucket`):

| Bucket | Mean novelty | Mean depth | Count |
| --- | ---: | ---: | ---: |
| Q1 | 1.8181 | 5.5047 | 4139 |
| Q2 | 2.1294 | 5.2771 | 3868 |
| Q3 | 2.3759 | 4.7578 | 4025 |
| Q4 | 2.6994 | 4.4064 | 3981 |
| Q5 | 3.3374 | 3.6637 | 3955 |

## Position in word

- Effect size (`summary.json.position_in_word.effect_size`): 0.4611

| Category | Mean depth | Count |
| --- | ---: | ---: |
| first_after_space | 8.0000 | 2903 |
| other_word_start | 7.8242 | 620 |
| middle_of_word | 4.4676 | 8361 |
| end_before_boundary | 3.7599 | 3457 |
| non_word | 3.4474 | 4627 |

## Local entropy

- Spearman rho (`summary.json.local_entropy.effect_size`): -0.1171
- Sliding radius: 64

| Bucket | Mean entropy (bits) | Mean depth | Count |
| --- | ---: | ---: | ---: |
| Q1 | 4.1570 | 5.1883 | 3994 |
| Q2 | 4.2964 | 5.0713 | 3995 |
| Q3 | 4.4054 | 4.7387 | 3992 |
| Q4 | 4.5546 | 4.4289 | 3994 |
| Q5 | 4.7179 | 4.2049 | 3993 |

## After punctuation

- Point-biserial r (`summary.json.after_punctuation.effect_size`): -0.2312
- Mean depth after punctuation: 1.0789
- Mean depth otherwise: 4.8936
- Mean depth difference: -3.8148

## Word frequency

- Spearman rho (`summary.json.word_frequency.effect_size`): 0.1607
- Word-character tokens analyzed: 15341

| Bucket | Mean rarity | Mean depth | Count |
| --- | ---: | ---: | ---: |
| Q1 | 2.6259 | 4.5431 | 3121 |
| Q2 | 3.2295 | 4.3358 | 3124 |
| Q3 | 3.8206 | 5.2400 | 2988 |
| Q4 | 4.7523 | 5.7377 | 6108 |

## Direct findings

- Strongest effect by the script's ranking is **position_in_word** at 0.4611 (`summary.json.effect_ranking[0]`).
- Weakest requested effect is **local_entropy** at 0.1171 (`summary.json.effect_ranking[-1]`).
- Bigram novelty is negative in this reproduction: depth moves from 5.5047 in the lowest-novelty bucket to 3.6637 in the highest-novelty bucket (`summary.json.bigram_novelty.mean_depth_by_bucket`).
- Local entropy is also negative here: depth moves from 5.1883 in the lowest-entropy bucket to 4.2049 in the highest-entropy bucket (`summary.json.local_entropy.mean_depth_by_bucket`).
- Immediate post-punctuation positions are shallower in this reproduction: 1.0789 after punctuation vs 4.8936 elsewhere (`summary.json.after_punctuation`).
- Rare-word membership is directional but modest: rho = 0.1607, with mean depth 4.5431 in the most frequent bucket and 5.7377 in the rarest bucket (`summary.json.word_frequency`).
