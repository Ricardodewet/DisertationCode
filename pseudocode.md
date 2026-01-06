# Pseudocode Breakdown of the Adaptive Assessment Project

This document provides a high‑level, language‑agnostic description of how the system works.  It is intended to guide readers through the logic without exposing them to specific programming syntax.  The pseudocode uses simple control structures (loops, conditionals) and descriptive names to clarify intent.

## 1. Simulated Learner Environment

The environment models a learner interacting with questions of varying difficulty.  It maintains hidden states (ability and engagement) and exposes observable signals.  A typical episode unfolds as follows:

```
FUNCTION ResetEnvironment(max_steps, seed):
    Set random seed
    ability ← Sample normal distribution with mean 1.0 and st.dev. 0.3, clipped to [0, 2]
    engagement ← 1.0
    last_action ← None
    last_correct ← 0
    step_count ← 0
    RETURN ConstructState(ability, engagement, last_action, last_correct)

FUNCTION Step(action):
    difficulty ← Map action to numeric level (0.0, 1.0 or 2.0)
    probability_correct ← Logistic(ability − difficulty)
    correct ← Draw a Bernoulli outcome with this probability
    response_time ← 1.0 + 0.5 × |difficulty − ability|
    IF correct THEN
        ability ← Clip(ability + 0.05, 0, 2)
    ELSE
        ability ← Clip(ability − 0.02, 0, 2)
    END IF
    engagement ← Clip(engagement − 0.1 × |difficulty − ability| + (0.05 if correct), 0, 1)
    reward ← correct + (1 − |difficulty − ability| / 2) − response_time / 3
    last_action ← action
    last_correct ← correct
    step_count ← step_count + 1
    done ← (step_count ≥ max_steps)
    RETURN ConstructState(ability, engagement, last_action, last_correct), reward, done, {correct, response_time, ability, engagement}

FUNCTION ConstructState(ability, engagement, last_action, last_correct):
    ability_norm ← ability / 2
    one_hot_action ← [0, 0, 0]
    IF last_action is not None THEN
        one_hot_action[last_action] ← 1
    END IF
    RETURN [ability_norm, engagement] + one_hot_action + [last_correct]
```

The `ResetEnvironment` function is called at the beginning of each episode.  The `Step` function updates the learner state based on the selected difficulty and produces the next state, reward and auxiliary information.  The `ConstructState` helper builds the vector passed to the agent.

## 2. PPO Agent Training Loop

Training proceeds over multiple episodes and seeds.  The agent collects experience, computes advantage estimates and updates its parameters.  The key functions are summarised below.

```
FOR each seed IN list_of_seeds DO
    Initialise environment with max_steps and seed
    Initialise PPO agent with state_dim and action_dim
    FOR episode FROM 1 TO num_episodes DO
        state ← ResetEnvironment()
        Initialise empty lists: states, actions, log_probs, values, rewards, dones
        done ← False
        WHILE not done DO
            action, log_prob, value ← agent.Act(state)
            next_state, reward, done, info ← Step(action)
            Append state, action, log_prob, value, reward, done to lists
            state ← next_state
        END WHILE
        returns, advantages ← agent.ComputeReturnsAndAdvantages(rewards, values, dones)
        agent.Update(states, actions, log_probs, returns, advantages)
        Log each step to a CSV file with columns: episode, step, action, difficulty, ability, engagement, response_time, correct, reward, prob_correct, seed, method
    END FOR
END FOR
```

The `Act` function computes action probabilities via a softmax over linear scores and samples an action.  `ComputeReturnsAndAdvantages` applies discounted returns and Generalised Advantage Estimation (GAE).  `Update` performs multiple gradient‑descent passes with the PPO objective:

```
FUNCTION agent.Update(states, actions, old_log_probs, returns, advantages):
    FOR epoch FROM 1 TO num_epochs DO
        Shuffle indices of the collected samples
        FOR each mini_batch DO
            Compute current log_probs and values for batch states and actions
            ratio ← exp(new_log_probs − old_log_probs)
            surrogate1 ← ratio × advantages
            surrogate2 ← Clip(ratio, 1 − eps_clip, 1 + eps_clip) × advantages
            policy_loss ← Negative mean of the minimum of surrogate1 and surrogate2
            value_loss ← 0.5 × mean((values − returns)²)
            Total loss ← policy_loss + value_coef × value_loss
            Compute gradients of loss w.r.t. policy and value parameters
            Update weights by subtracting learning_rate × gradients
        END FOR
    END FOR
```

Note that entropy regularisation is omitted in this simplified implementation.  After updating the agent, the loop begins a new episode.

## 3. Baseline Policy Execution

Baseline policies do not learn; they implement fixed rules.  The general execution template is:

```
FOR each baseline_name IN ["random", "static", "heuristic", "staircase"] DO
    FOR each seed IN list_of_seeds DO
        Initialise environment with max_steps and seed
        IF baseline_name == "staircase" THEN
            current_action ← 1 (start at Medium)
        END IF
        FOR episode FROM 1 TO num_episodes DO
            state ← ResetEnvironment()
            done ← False
            WHILE not done DO
                SELECT action based on the baseline policy:
                    random: pick 0, 1 or 2 uniformly
                    static: pick 1 always
                    heuristic: pick 0 if ability < 0.7; 1 if ability ∈ [0.7, 1.3); 2 otherwise
                    staircase: use current_action; after each step update current_action by +1 on correct, −1 on incorrect (bounded in [0, 2])
                next_state, reward, done, info ← Step(action)
                Log step information as for PPO
            END WHILE
        END FOR
    END FOR
END FOR
```

This produces logs analogous to those generated during PPO training, but without any parameter updates.

## 4. Metrics Aggregation

After all runs have been logged, they are summarised into per‑run metrics:

```
FUNCTION SummariseRuns(run_directory):
    FOR each CSV file in run_directory DO
        Read data into a table
        mean_reward ← Average of the reward column
        accuracy ← Average of the correct column (proportion of 1s)
        mean_engagement ← Average engagement over all steps
        mean_response_time ← Average response_time over all steps
        learning_gain ← Last ability value minus first ability value
        episodes ← Number of distinct episode indices
        Append a record with method, seed and computed metrics
    END FOR
    RETURN Table of all records
```

The resulting table is saved as `run_metrics.csv` and forms the basis for statistical analysis.

## 5. Statistical Tests

The comparison between PPO and each baseline uses Welch’s t‑test and Cohen’s *d* effect size for each metric:

```
FUNCTION ComputeStatistics(metrics_table, reference_method):
    descriptive_stats ← Compute mean and standard deviation of each metric grouped by method
    inferential_stats ← Empty table
    FOR each method ≠ reference_method DO
        FOR each metric DO
            x ← Values of metric for reference_method
            y ← Values of metric for current method
            t_statistic, p_value ← WelchTTest(x, y)
            effect_size ← CohenD(x, y)
            Record t_statistic, p_value and effect_size for (method, metric)
        END FOR
    END FOR
    RETURN descriptive_stats, inferential_stats
```

The descriptive table and inferential table are written to CSV files.  These tables allow you to judge whether differences in mean reward, accuracy, learning gain or engagement are statistically significant.

## 6. Plot Generation

Finally, various plots are created from the raw logs to visualise learning curves and behaviour distributions:

```
FUNCTION PlotResults(run_directory, output_directory):
    FOR each metric_name IN ["reward", "accuracy", "engagement"] DO
        curves ← Extract per‑episode mean metric per method and seed
        Compute mean and standard deviation across seeds
        Plot episode index on the x‑axis and mean metric on the y‑axis with confidence bands
        Save as PNG with an appropriate name
    END FOR
    difficulties ← Compute histograms of chosen difficulties binned by ability for each method
    FOR each method DO
        Plot a bar chart showing how often Easy, Medium and Hard were selected in each ability bin
    END FOR
    Save difficulty distribution plot
```

## 7. End‑to‑End Pipeline Summary

Putting it all together, running the project involves the following high‑level sequence:

```
1. Initialise experiment by editing the YAML configuration.
2. Train the PPO agent with scripts/train.py for each seed; write logs.
3. Execute baseline policies with scripts/baselines.py; write logs.
4. Aggregate logs into per‑run summaries with scripts/evaluate.py.
5. Perform statistical analyses with scripts/stats.py.
6. Generate plots with scripts/plot_results.py.
7. Review report_summary.md to interpret results and consult ASSUMPTIONS.md for model details.
```

This pseudocode encapsulates the logical flow of the adaptive assessment project.  It is structured so that readers unfamiliar with Python can grasp how the system operates and how the pieces fit together.