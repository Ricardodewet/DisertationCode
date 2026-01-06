# Experimental Results Summary

This document summarises the outcomes of the simulation experiments described
in the thesis.  A Proximal Policy Optimisation (PPO) agent was compared
against four baseline strategies—Random selection, Static Medium, Heuristic
ZPD and Staircase—across three random seeds.  Each run consisted of 100
episodes of 20 interactions.  The metrics reported include average reward,
accuracy, learning gain, engagement and response time.  Welch’s t‑tests
(reference = PPO) were used for inferential comparisons and Cohen’s *d*
quantified effect sizes.

## Descriptive Statistics

| Method     | Mean Reward | Accuracy | Learning Gain | Mean Engagement | Mean Response Time |
|-----------:|------------:|---------:|--------------:|----------------:|-------------------:|
| **PPO**    | 0.69 ± 0.10 | 0.54 ± 0.03 | 0.51 ± 0.29 | 0.48 ± 0.01 | 1.30 ± 0.01 |
| **Random** | 0.68 ± 0.12 | 0.54 ± 0.02 | 0.63 ± 0.08 | 0.47 ± 0.01 | 1.30 ± 0.01 |
| **Static** | 0.97 ± 0.03 | 0.55 ± 0.02 | 0.63 ± 0.14 | 0.86 ± 0.02 | 1.10 ± 0.02 |
| **Heuristic** | 0.92 ± 0.04 | 0.46 ± 0.02 | 0.42 ± 0.02 | 0.85 ± 0.02 | 1.11 ± 0.02 |
| **Staircase** | 0.78 ± 0.03 | 0.52 ± 0.03 | 0.53 ± 0.12 | 0.62 ± 0.01 | 1.21 ± 0.01 |

Values are reported as mean ± standard deviation across seeds.  Mean reward
combines correctness, flow alignment and response time (see
`ASSUMPTIONS.md` for details).

## Inferential Statistics

Welch’s t‑tests compared the PPO agent against each baseline.  A negative
t‑statistic indicates that the baseline outperformed PPO on the given
metric.  The table below lists the t statistic, two‑sided *p*‑value and
Cohen’s *d* effect size (larger absolute values indicate stronger effects).

| Baseline vs PPO | Metric | t Statistic | p‑value | Cohen’s *d* |
|----------------:|-------:|-----------:|--------:|------------:|
| **Random** | Mean Reward | 0.30 | 0.65 | 0.40 |
| | Accuracy | 0.31 | 0.64 | 0.41 |
| | Learning Gain | 0.68 | 0.53 | 0.91 |
| | Engagement | –0.88 | 0.38 | –1.14 |
| **Static** | Mean Reward | –31.0 | <0.001 | –21.95 |
| | Accuracy | –2.64 | 0.06 | –3.65 |
| | Learning Gain | –0.50 | 0.64 | –0.71 |
| | Engagement | –18.97 | <0.001 | –19.35 |
| **Heuristic** | Mean Reward | –23.57 | <0.001 | –27.57 |
| | Accuracy | –7.73 | 0.004 | –10.91 |
| | Learning Gain | –2.49 | 0.07 | –3.52 |
| | Engagement | –21.04 | <0.001 | –21.83 |
| **Staircase** | Mean Reward | –10.04 | <0.001 | –14.75 |
| | Accuracy | –1.41 | 0.23 | –1.98 |
| | Learning Gain | –0.12 | 0.92 | –0.16 |
| | Engagement | –10.89 | <0.001 | –11.64 |

## Conclusions

Across the tested scenarios the PPO agent did not outperform the simpler
baseline strategies on most metrics.  Static and heuristic baselines achieved
higher mean rewards, accuracy and engagement.  Welch’s t‑tests confirmed
large, statistically significant differences between PPO and these baselines
(*p* < 0.001) with very large effect sizes (|*d*| > 10).  Even the Random
and Staircase policies performed comparably to PPO.  These results suggest
that in the present simulation the linear PPO agent failed to learn an
effective difficulty adjustment policy, likely due to the simplicity of the
policy model and limited training horizon.  The static medium difficulty
consistently kept the learner within the flow zone (ability ≈ difficulty),
yielding high engagement and reward.

While disappointing, these findings highlight important limitations of this
simulation: the learner model is highly idealised, the PPO implementation
uses a single linear layer without non‑linearities and the number of
episodes is modest.  Real‑world adaptive systems may benefit from more
expressive policies (e.g. multi‑layer networks), longer training and richer
state representations.  Nevertheless, the framework demonstrates how to
construct reproducible experiments, collect detailed logs, perform
statistical analysis and generate publication‑quality figures, laying the
groundwork for more sophisticated future studies.
