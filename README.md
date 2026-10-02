<div align="center">

# Safe Driving with Deep Q-Networks in MetaDrive
### Autonomous Driving Agent with Zero-Shot Generalization Across Procedural Scenarios

</div>

## Overview

This repository implements an autonomous driving agent trained with **Deep Q-Networks (DQN)** in the [MetaDrive](https://github.com/metadriverse/metadrive) simulation environment. The vehicle receives a continuous 259-dimensional sensory observation (including 240 LiDAR detection beams and ego-state dynamics) and navigates procedural road topologies with dense traffic.

Although trained solely on **20 procedural maps**, the agent demonstrates strong **zero-shot generalization**, navigating **100 completely unseen procedural maps** without any fine-tuning or prior exposure.

---

## Benchmark Results

### Zero-Shot Generalization (100 Unseen Maps)
Evaluated greedily ($\epsilon = 0$) across 100 consecutive unseen map seeds (`seed=21` to `seed=120`):

| Metric | Result | Target / Baseline |
| :--- | :---: | :---: |
| **Success Rate (Arrived at Destination)** | **42.0%** | > 30% |
| **Average Episode Reward** | **332.74** | > 0.0 |
| **Vehicle Crash Rate** | **34.0%** | Balanced |
| **Out-of-Road Rate** | **24.0%** | Balanced |
| **Average Episode Length** | **302.3 steps** | Decisive navigation |

> Detailed episode logs and metrics are stored in [`Evaluation/evaluation_report.json`](Evaluation/evaluation_report.json).

---

## Training Dynamics

The agent was trained for **10,000 episodes** across 20 distinct procedural environments:

<p align="center">
  <img src="docs/training_plots.png" alt="DQN Training Dynamics Across 10,000 Episodes" width="100%">
</p>

- **Reward Progression:** Average episode reward transitioned smoothly from initial negative penalties up to ~400, reflecting balanced speed and safe navigation.
- **Episode Stability:** Step counts converged to the 250–400 step regime, completely overcoming stalling failure modes.
- **Outcome Convergence:** Out-of-road failures dropped sharply, while destination arrival rate climbed steadily.

---

## Key Design Decisions

To overcome common physical vehicle RL failure modes (such as reckless maximum throttling or penalty-avoiding paralysis):
- **Forward-Forced Action Space:** Eliminates zero and negative throttle choices; minimum throttle is fixed to `0.1` (engine braking), preventing the agent from parking to exploit penalty structures.
- **Balanced Reward Shaping:** Destination arrival bonus (`+200.0`) with balanced penalties (`-50.0`) for collisions and out-of-road events, encouraging decisive forward progress.
- **Architecture & Scaling:** Expanded network capacity to 512 hidden units across 3 layers and trained for 10,000 episodes with periodic target network synchronization every 40 episodes.

---

## Technical Architecture

### State & Action Representation
- **Observation Space:** Flattened **259-dimensional vector** (240 LiDAR rangefinder rays + 19 ego-vehicle telemetry features).
- **Action Space:** 6 discrete steering and throttle maneuvers:

| Action Index | Steering | Throttle | Semantic Driving Intent |
| :---: | :---: | :---: | :--- |
| **0** | `-1.00` | `+0.10` | Hard Left + Engine Brake (Emergency avoidance) |
| **1** | `-0.75` | `+0.60` | Mild Left + Forward (High-speed cornering) |
| **2** | ` 0.00` | `+0.10` | Straight + Engine Brake (Traffic deceleration) |
| **3** | ` 0.00` | `+0.80` | Straight + Full Gas (Speed accumulator) |
| **4** | `+1.00` | `+0.10` | Hard Right + Engine Brake (Emergency avoidance) |
| **5** | `+0.75` | `+0.60` | Mild Right + Forward (High-speed cornering) |

### DQN Network & Hyperparameters
The policy network uses a 3-layer Multi-Layer Perceptron (MLP) trained with Adam and Mean Squared Error loss against a frozen target network:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Architecture** | `259 → 512 → 512 → 6` | Fully connected with ReLU activations |
| **Replay Buffer Capacity** | `100,000` | Uniform experience replay memory |
| **Mini-batch Size** | `256` | Number of transition tuples per gradient update |
| **Discount Factor ($\gamma$)** | `0.995` | Emphasizes long-horizon navigational rewards |
| **Optimizer & Learning Rate** | Adam, $\alpha = 3 \times 10^{-4}$ | Adaptive moment estimation |
| **Exploration ($\epsilon$)** | $1.0 \to 0.05$ | Epsilon-decay factor of $0.9995$ per episode |
| **Target Sync Frequency** | Every 40 episodes | Hard parameter copy $\theta_{\text{target}} \leftarrow \theta_{\text{policy}}$ |

---

## Repository Structure

```
├── main.py                     # Training entry point (10k episodes, checkpointing)
├── evaluate.py                 # Zero-shot evaluation across 100 unseen maps
├── visualize.py                # MetaDrive 3D/2D graphical renderer
├── requirements.txt            # Environment dependencies
├── README.md                   # Project documentation & benchmark analysis
├── src/
│   ├── __init__.py
│   ├── agent.py                # DQNAgent class (epsilon-greedy, Bellman updates)
│   ├── network.py              # DQNNetwork PyTorch architecture
│   ├── env_utils.py            # MetaDrive configuration, observation & action mapping
│   └── replay_buffer.py        # Optimized experience replay buffer
├── docs/
│   ├── report.ipynb            # Interactive research report & data analysis
│   └── training_plots.png      # High-resolution training dynamics figure
├── Evaluation/
│   └── evaluation_report.json  # Serialized metrics from zero-shot benchmark
└── models/
    ├── dqn_trained.pt          # Final trained PyTorch model weights (1.5 MB)
    ├── rewards_history.npy     # Episode-by-episode reward history (10k eps)
    ├── steps_history.npy       # Episode length history (10k eps)
    └── crash_history.npy       # Termination outcome history (0=succ, 1=oor, 2=crash)
```

---

## Installation & Setup

### Prerequisites
- **Python:** $\ge 3.8$ and $< 3.12$ (MetaDrive simulator requires Python $< 3.12$).
- **OS:** Windows, Linux, or macOS.
- **GPU (Optional):** PyTorch automatically utilizes CUDA if available, falling back to CPU.

### Step-by-Step Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ParhamM83/DQN-Metadrive-Driving-Agent.git
   cd DQN-Metadrive-Driving-Agent
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (Command Prompt / PowerShell)
   python -m venv venv
   .\venv\Scripts\activate

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Usage Guide

### 1. Visualize Trained Policy
Render the trained agent navigating unseen test scenarios in real-time:

```bash
# 3D third-person view (Panda3D window)
python visualize.py --mode 3D --episodes 5 --seed 21 --scenarios 100

# Top-down 2D bird's-eye view
python visualize.py --mode top_down --episodes 5 --seed 21 --scenarios 100
```

### 2. Zero-Shot Evaluation
Run greedy evaluation on 100 unseen procedural maps:

```bash
python evaluate.py --episodes 100 --seed 21 --scenarios 100
```
This prints the outcome distribution table to the console and saves the results to `Evaluation/evaluation_report.json`.

### 3. Training From Scratch
To launch a complete training run:

```bash
python main.py --episodes 10000 --seed 1 --scenarios 20 --batch-size 256
```
- Automatically creates `models/` directory and saves checkpoints every 1,000 episodes.
- Safe under `Ctrl+C` (saves progress cleanly before termination).
