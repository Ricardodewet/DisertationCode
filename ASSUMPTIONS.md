# Assumptions and Design Decisions

This document lists the principal assumptions made when implementing the
simulation environment, reward function, learner model and reinforcement
learning algorithms.  The thesis (`main.tex`) provides high‑level
descriptions but omits many concrete specifications.  To deliver a
functional experimental pipeline, several defensible modelling choices were
made.  If future work uncovers better alternatives, these assumptions can
easily be revised in the code.

## Learner Model

* **Ability scale** – The learner’s latent ability is represented on a
  continuous scale `[0, 2]`, with 0 corresponding to the easiest content and
  2 corresponding to the hardest.  The discrete difficulty levels (Easy,
  Medium, Hard) are mapped to numeric values 0.0, 1.0 and 2.0 respectively.

* **Initial ability** – Each simulated learner begins with an ability
  sampled from a normal distribution with mean 1.0 and standard deviation
  0.3, truncated to the range `[0, 2]`.  Different random seeds yield
  different initial abilities across runs.

* **Probability of correct** – The probability that the learner answers
  correctly is modelled by a logistic function:
  
  \[P(\text{correct}) = \frac{1}{1 + \exp(-(\alpha - d))}\]

  where \(\alpha\) is the learner’s current ability and \(d\) is the numeric
  value of the chosen difficulty.  This choice approximates Item Response
  Theory for multiple difficulty levels.

* **Ability update** – After each interaction the ability evolves:
  
  * If the learner answers correctly, ability increases by `0.05`.
  * If the learner answers incorrectly, ability decreases by `0.02`.
  * Ability is clipped to `[0, 2]` after the update.

  This simple rule captures the notion that success promotes mastery whereas
  failure results in minor forgetting or frustration.

* **Engagement update** – Engagement is a proxy for Flow/ZPD alignment.
  Initial engagement is set to `1.0`.  After each step:
  
  * Engagement decreases by `0.1 * |d - \alpha|` (punishing large
    mismatches between difficulty and ability).
  * Engagement increases by `0.05` if the learner answers correctly.
  * Engagement is clipped to `[0, 1]`.

  The agent is thus incentivised to keep the learner within an optimal
  challenge zone where `|d - ability|` is small.

* **Response time** – The time taken to respond is simulated as
  
  \[t_{response} = 1.0 + 0.5 \times |d - \alpha|\]

  representing slower responses when the question is too easy or too hard.

## State Representation

The reinforcement learning agent receives a normalised state vector:

1. **Ability estimate** – The current ability scaled to `[0, 1]` by dividing
   by the maximum ability (2.0).
2. **Engagement** – Already in `[0, 1]`.
3. **Last action** – The previously chosen difficulty encoded as a one‑hot
   vector of length 3 (Easy, Medium, Hard).  For the first timestep this is
   all zeros.
4. **Last correctness** – A binary indicator (1 for correct, 0 otherwise).

Concatenating these features yields a 7‑dimensional state vector.

## Reward Function

The reward for each step is a weighted sum of three components:

1. **Correctness reward** (`r_c`) – `+1` if the learner answers correctly,
   `0` otherwise.
2. **Flow/ZPD bonus** (`r_f`) – `1 - |d - \alpha| / 2.0`.  This term is
   maximal when difficulty matches ability and decreases linearly as the
   mismatch grows.
3. **Response time penalty** (`r_t`) – `- t_{response} / 3.0` to penalise
   slow responses.

The total reward is `r = r_c + r_f + r_t`.  This design encourages the
agent to choose difficulties that are challenging yet achievable, to produce
quick answers and to maximise correctness.

## Reinforcement Learning Algorithm

A minimalist Proximal Policy Optimisation (PPO) implementation was
constructed using only `numpy`.  Key design choices include:

* **Policy and value networks** – Both are single‑layer linear models
  mapping the 7‑dimensional state to 3 action logits (policy) and a single
  state value.  Non‑linearities are omitted for simplicity; however the
  clipping mechanism and advantage normalisation still enable learning.
* **Generalised Advantage Estimation** – Advantages are computed with
  discount factor `gamma=0.99` and trace decay `lambda=0.95`.
* **Clipping** – Policy ratios are clipped to `[1-0.2, 1+0.2]` during
  optimisation.
* **Learning rate and coefficients** – A learning rate of `0.01` is used.
  The value loss coefficient is `0.5` and entropy regularisation is not
  employed (empirically unnecessary for this small task).  Each training
  update uses 3 epochs over minibatches of 64 samples.

## Baseline Policies

Several baseline strategies were implemented to benchmark the PPO agent:

1. **Random** – Selects a difficulty uniformly at random each step.
2. **Static Medium** – Always chooses the Medium difficulty (1.0).
3. **Heuristic ZPD** – If the learner’s estimated ability is < 0.7 choose
   Easy; if between 0.7 and 1.3 choose Medium; otherwise choose Hard.
4. **Staircase** – Starts at Medium; increases one level after a correct
   answer; decreases one level after an incorrect answer; bounds at Easy/Hard.

These rules approximate the types of baseline models described in the
literature review: random policies, static adaptive testing and rule‑based
ZPD alignment.

## Episode Configuration

Each episode consists of 20 interactions (questions).  Experiments run
multiple episodes per seed (default 200).  After each episode the log of
states, actions, rewards, correctness, response times and ability values is
stored in CSV format.  Aggregated metrics are computed from these logs.

## Metrics and Statistical Tests

Metrics are calculated per run and aggregated across seeds.  Inferential
statistics use Welch’s t‑tests comparing PPO against each baseline on
accuracy, learning gain, engagement and reward.  Cohen’s *d* is reported as
an effect size.  Where normality assumptions are violated, non‑parametric
alternatives (e.g. Mann–Whitney *U*) can be substituted.
