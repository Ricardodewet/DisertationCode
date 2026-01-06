# Project Breakdown and Usage Guide

This document provides a comprehensive yet accessible explanation of the reinforcement‑learning adaptive assessment project. Its purpose is to enable any reader to understand the rationale, architecture and execution flow of the system, as well as the steps needed to run and test it on their own machine.

## 1. Overview

The project simulates an **adaptive educational assessment** in which a software agent selects the difficulty of each question presented to a learner.  The goal is to personalise the level of challenge so as to maximise learning gains while maintaining engagement.  To achieve this, the system models a virtual learner with latent *ability* and *engagement* states, uses a reinforcement‑learning algorithm (Proximal Policy Optimisation, PPO) to learn a difficulty‑selection policy, and compares its performance against several heuristic baseline strategies.  The output consists of detailed logs of learner–agent interactions, aggregated performance metrics, statistical comparisons and visualisations.

The repository is organised into distinct subfolders:

| Folder             | Purpose |
|--------------------|---------|
| `src/`             | Core modules, including the simulated environment (`envs/`), the PPO agent and baseline policies (`agents/`), evaluation helpers (`evaluation/`) and utilities (`utils/`). |
| `scripts/`         | Command‑line entry points for running the simulation pipeline (training, baselines, evaluation, statistics, plotting). |
| `configs/`         | YAML files specifying hyperparameters such as the number of episodes, seeds and learning rates. |
| `outputs/`         | Generated data: raw interaction logs (`runs/`), aggregated tables (`tables/`) and figures (`figures/`). |
| `tests/`           | PyTest unit and integration tests ensuring the correctness of each component. |
| `ASSUMPTIONS.md`   | Explanation of modelling choices (state variables, reward design, learner dynamics). |
| `report_summary.md`| Plain‑English summary of the experimental results and conclusions. |

## 2. Conceptual Components

### 2.1 Simulated Environment

At the heart of the project is a simplified model of learner behaviour (`src/envs/learner_env.py`).  Each episode represents a series of interactions between a learner and the assessment system.  The environment maintains two hidden variables:

1. **Ability** — a continuous value indicating the learner’s proficiency on a scale from 0 (novice) to 2 (expert).  The probability of answering a question correctly increases as the chosen difficulty approaches the learner’s ability.
2. **Engagement** — a proxy for the learner’s motivational state, influenced by how well the difficulty matches ability.  Engagement decreases when questions are too easy or too hard and increases when performance improves.

On each step, the agent chooses a difficulty level (Easy, Medium or Hard).  The environment returns observable signals: correctness (0 or 1), response time (longer for mismatched difficulties), current engagement and a reward that combines these factors.  The learner’s ability and engagement evolve after each answer to reflect learning progress and boredom/frustration.

### 2.2 Reinforcement‑Learning Agent (PPO)

The PPO agent (`src/agents/ppo_agent.py`) learns to select difficulties based on the current state.  Its policy and value function are represented by linear models that map the state vector to action probabilities and an estimated return, respectively.  During training, the agent collects trajectories of states, actions, rewards and values, computes advantages using Generalised Advantage Estimation and updates its parameters by maximising the clipped PPO objective.  Although minimalist compared with deep neural networks, this implementation captures the core algorithmic ideas.

### 2.3 Baseline Policies

To contextualise the performance of the PPO agent, several non‑learning strategies are implemented in `src/agents/baseline_policies.py`:

* **Random** — chooses a difficulty uniformly at random.  Serves as a lower‑bound reference.
* **Static Medium** — always selects the medium difficulty.  Tests the effect of never adapting.
* **Heuristic ZPD** — selects Easy, Medium or Hard based on simple thresholds on the learner’s ability estimate.  Mimics a rule‑based Zone of Proximal Development strategy.
* **Staircase** — starts at Medium and raises or lowers difficulty after correct or incorrect answers, akin to classical computerised adaptive testing.

### 2.4 Evaluation and Visualisation

Interaction logs stored in `outputs/runs/` are aggregated by `scripts/evaluate.py` into per‑run statistics (accuracy, average reward, learning gain, engagement and response time).  `scripts/stats.py` then computes descriptive statistics and Welch’s t‑tests comparing PPO to each baseline, reporting p‑values and effect sizes.  Finally, `scripts/plot_results.py` generates plots summarising reward trajectories, accuracy and engagement trends, and the distribution of difficulty choices relative to the learner’s ability.

## 3. Running the Project

Follow these steps to execute the pipeline from scratch.  Commands assume you are in the project root.

1. **Setup a Python environment (optional but recommended)**.  Create and activate a virtual environment, then install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate.bat
   pip install numpy pandas matplotlib scipy pyyaml
   # For running tests:
   pip install pytest
   ```

2. **Configure the experiment**.  The YAML file `configs/experiment.yaml` defines key parameters:

   ```yaml
   environment:
     max_steps: 20       # number of questions per episode
   training:
     episodes: 100        # number of episodes per run
     seeds: [0, 1, 2]     # random seeds for reproducibility
     learning_rate: 0.01
     gamma: 0.99
     lam: 0.95
     eps_clip: 0.2
     value_coef: 0.5
     epochs: 3
     batch_size: 64
   ```

   Adjust these values to explore different scenarios (e.g. more episodes or alternative seeds).

3. **Train the PPO agent**.  Run:

   ```bash
   python -m scripts.train --config configs/experiment.yaml --out outputs/runs
   ```

   This trains the PPO agent for each seed specified in the configuration and writes a CSV log for every seed into `outputs/runs/`.

4. **Run baseline policies**.  To benchmark against heuristic strategies, execute:

   ```bash
   python -m scripts.baselines --config configs/experiment.yaml --out outputs/runs
   ```

   This produces additional CSV logs for each baseline and seed.

5. **Aggregate metrics**.  Summarise raw logs by running:

   ```bash
   python -m scripts.evaluate --run_dir outputs/runs --out outputs/tables
   ```

   The resulting `run_metrics.csv` in `outputs/tables/` contains one row per run with calculated metrics.

6. **Compute statistics**.  To compare methods and determine statistical significance, run:

   ```bash
   python -m scripts.stats --input outputs/runs --out outputs/tables
   ```

   This writes `descriptive_stats.csv` and `inferential_stats.csv` to `outputs/tables/`, summarising mean/standard deviations and t‑tests versus PPO.

7. **Generate plots**.  For visualisations, execute:

   ```bash
   python -m scripts.plot_results --input outputs/runs --out outputs/figures
   ```

   Figures such as reward curves, accuracy trends and difficulty distributions are saved in `outputs/figures/`.  These are suitable for inclusion in reports or presentations.

8. **Review results**.  Consult `report_summary.md` for a narrative of the outcomes.  The summary articulates which methods performed best, interprets the statistical tests and discusses limitations of the simulation.

9. **Run unit tests** (optional).  To validate individual modules, install `pytest` and run:

   ```bash
   pytest -q
   ```

   The tests verify environment dynamics, agent behaviour, baseline policies and that the pipeline can run end to end in a simplified setting.

## 4. Interpretation of Outputs

* **Raw logs (`outputs/runs/`)** — Each CSV contains step‑by‑step traces for one method and seed.  Columns include episode and step numbers, chosen difficulty, ability, engagement, response time, correctness, reward and estimated probability of correctness.  These logs can be used for in‑depth analysis or debugging.

* **Run metrics (`outputs/tables/run_metrics.csv`)** — Aggregated metrics per method and seed.  Key fields include:
  * `mean_reward`: average reward per step.
  * `accuracy`: proportion of correct answers.
  * `learning_gain`: change in ability from the first to the last step.
  * `mean_engagement`: average engagement level.
  * `mean_resp_time`: average response time.

* **Descriptive statistics (`outputs/tables/descriptive_stats.csv`)** — Mean and standard deviation of each metric aggregated over all seeds for each method.

* **Inferential statistics (`outputs/tables/inferential_stats.csv`)** — For each baseline method, the table lists the Welch t‑statistic, p‑value and Cohen’s d effect size when comparing its performance to the PPO agent.  These values help determine whether observed differences are statistically meaningful and how large they are.

* **Figures (`outputs/figures/`)** — Graphical summaries.  Reward curves depict how agents improve over episodes; accuracy and engagement curves track learner success and motivation; difficulty distribution charts illustrate how often each difficulty level was chosen across ability bins.

## 5. Common Pitfalls and Tips

* **Activating the virtual environment** — Ensure the `.venv` is activated before installing dependencies or running scripts.  Do not `cd` into `.venv`; instead, call `source .venv/bin/activate` (or the appropriate Windows command) from the project root.

* **Consistent seeding** — The configuration’s `seeds` list determines reproducibility.  Modify or extend this list to perform robustness checks.

* **Experiment scaling** — Increasing `episodes` or using more expressive models (e.g. multi‑layer neural networks) will demand more computation.  Monitor training time and memory usage when scaling up.

* **Interpreting results** — In the provided simulation, some baselines may outperform the PPO agent because the agent’s linear model is intentionally simple and the reward strongly favours moderate challenges.  When reporting findings, emphasise that the main contribution is the reproducible pipeline rather than absolute superiority of PPO.

This breakdown should equip you to navigate, run and understand the entire project.  Consult the code and the `ASSUMPTIONS.md` file for deeper details on specific modelling decisions.