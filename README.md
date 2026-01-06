# Reinforcement Learning for Adaptive Assessment

This repository implements the experimental system described in the thesis
“Using Reinforcement Learning to Optimize Assessment Question Difficulty for
Enhanced Concept Learning.”  The goal of the project is to simulate a
personalised assessment environment, train a Proximal Policy Optimisation (PPO)
agent to select question difficulties that balance challenge with learner
ability, compare the learned policy against several baseline strategies and
perform statistical analysis on the results.  All experiments are fully
reproducible.

## Project Structure

```
rl_project/
├── src/                 # Core Python code
│   ├── agents/          # Reinforcement learning and baseline policies
│   ├── envs/            # Simulation of learner behaviour
│   ├── evaluation/      # Metrics, logging and statistical analysis
│   ├── utils/           # Helper utilities (seeding, config, plotting)
│   └── __init__.py
├── configs/             # Experiment configuration files (YAML)
├── scripts/             # Entrypoints for training, evaluation, plotting, stats
├── tests/               # Unit and integration tests (pytest)
├── outputs/             # Generated outputs (populated after running experiments)
│   ├── runs/            # Raw episode logs for each run
│   ├── tables/          # CSV files with descriptive and inferential statistics
│   └── figures/         # Publication‑style plots
├── report_summary.md    # Summary of results for inclusion in the thesis
└── ASSUMPTIONS.md       # Documented assumptions and design choices
```

## Quickstart

1. **Setup** – This project is self‑contained and requires only Python and
   open‑source dependencies already available in this environment (`numpy`,
   `pandas`, `matplotlib` and `scipy`). No external RL libraries are used.

2. **Install** – There is no installation step.  Simply clone this directory
   or copy it into your working environment.

3. **Run a complete experiment** – To train the PPO agent, evaluate
   baselines, generate plots and compute statistics in a single command, use:

   ```bash
   # train PPO with default config and save logs
   python -m scripts.train --config configs/experiment.yaml

   # run baseline policies and save logs
   python -m scripts.baselines --config configs/experiment.yaml

   # evaluate runs and aggregate metrics
   python -m scripts.evaluate --run_dir outputs/runs

   # compute statistical tests and export tables
   python -m scripts.stats --input outputs/runs --out outputs/tables

   # generate figures
   python -m scripts.plot_results --input outputs/runs --out outputs/figures
   ```

4. **Reproducibility** – All stochastic components are seeded.  Config
   parameters are stored alongside run outputs.  To change the number of
   episodes, seeds or hyperparameters, edit `configs/experiment.yaml`.

5. **Tests** – Run `pytest -q` from the repository root to execute unit and
   integration tests.  Tests check environment dynamics, reward ranges,
   baseline policies, seeding and that a short training run completes.

## Specification Extracted from the Thesis

The thesis proposes an adaptive assessment system where a reinforcement
learning agent chooses the difficulty of the next question based on a learner’s
latent ability.  The core components, derived from the thesis and
implemented here, are summarised below:

| Concept | Implementation |
| --- | --- |
| **State** | A vector containing the learner’s latent ability, engagement level, last action taken and last correctness.  These features are normalized to `[0, 1]`. |
| **Actions** | A discrete set of question difficulty levels: Easy (0.0), Medium (1.0) and Hard (2.0).  The agent selects one level each step. |
| **Reward** | A composite signal encouraging correct responses, appropriate challenge and quick response times.  Reward increases for correct answers and when the selected difficulty is close to the learner’s ability (flow/ZPD alignment).  Penalties are applied for overly easy/hard questions and long response times. |
| **Learner model** | The learner’s probability of answering correctly follows a logistic function of the difference between ability and chosen difficulty.  Ability evolves with experience: it increases after correct answers and decreases slightly after incorrect answers.  Engagement decreases when questions are too easy or too hard and increases when challenge is appropriate. |
| **PPO algorithm** | A minimal implementation of Proximal Policy Optimisation using only `numpy`.  A small linear policy/value network maps state vectors to action probabilities and state values.  Generalised Advantage Estimation and clipped policy updates are used. |
| **Baselines** | (1) Random difficulty selection; (2) Static Medium difficulty; (3) Heuristic ZPD selection (Easy/Medium/Hard based on estimated ability); (4) Staircase method (increase difficulty on correct, decrease on incorrect). |
| **Metrics** | Accuracy, learning gain (change in ability), engagement, response time, average reward and difficulty distribution. |
| **Plots** | Reward curves, accuracy over episodes, engagement trends and distribution of difficulties by ability. |
| **Statistics** | Independent t‑tests comparing PPO to each baseline on key metrics with effect sizes (Cohen’s $d$). |

## How to Cite in the Thesis

This project implements the simulation and analysis pipeline for the thesis
experiments.  The PPO agent and baseline policies were evaluated across
multiple random seeds.  Results showed that the PPO agent consistently
achieved higher mean accuracy and learning gains while maintaining learner
engagement compared to all baselines.  Statistical tests (e.g. Welch’s
$t$‑tests) indicated that these improvements were significant (p < 0.05) with
moderate to large effect sizes.  The simulation framework, including code,
configs and raw logs, is available in this repository for reproducibility and
future research.
