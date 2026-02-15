# Jamb Neural Network (JAX Optimized)

An ultra-high-performance reinforcement learning agent for a complex Yamb (Jamb) variant, implemented using **JAX** and **Flax** for massive GPU parallelism.

## 🚀 Performance Highlights

- **Architecture:** Actor-Critic (PPO) with 3-layer MLP.
- **Speed:** ~650,000 steps per second on a single RTX 3090.
- **Top Score:** **2,044 points** (100k game eval).
- **Average Score:** **1,722 points** (Target: 1,750).

## 🧠 Training Strategy: "Crazy and Fast"

The breakthrough in performance came from a high-volume, low-overfitting approach:

- **Entropy Coefficient (0.03):** High exploration to prevent early convergence to local optima.
- **Low Epochs (4):** Prevents "memorizing" noise from lucky dice rolls.
- **Pure JAX Environment:** Entire game logic written in JAX to allow `vmap`-ing millions of games across GPU threads.

## 📁 Repository Structure

- `jamb_jax.py`: Core game engine and logic (JAX).
- `train_jax.py`: Training pipeline (PPO implementation).
- `evaluate_agent_100k.py`: Large-scale evaluation and report generation script.
- `watch_agent_jax.py`: Human-readable game replay tool.
- `models/`: Checkpoints from the best runs.
- `reports/`: Evaluation reports and score distributions.

## 🛠️ Requirements

- JAX (CUDA supported)
- Flax
- Distrax (for distributions)
- Optax (for optimization)
- Gym / Gymnasium

## 📊 Latest Results

The agent has successfully mastered complex game mechanics including:

- **Announcements:** Choosing when to commit to a column before rolling.
- **Column Constraints:** Navigating "Down", "Free", "Up", and "Anno" column rules.
- **Roll Management:** Strategically using the special 5-roll rule for the final turn.

Check out `jamb_crazyandfast_100k_report.md` for a detailed breakdown of cell averages and column completion speeds.
