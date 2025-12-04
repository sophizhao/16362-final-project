# 16-362 Final Project – Multi-Agent Goal Finding using Deep Reinforcement Learning

This repository contains our final project for **16-362 Mobile Robot Algorithms Laboratory**.

We study **multi-agent navigation** on a discrete 2D grid with static obstacles.  
Each agent has a start and a goal; the objective is to reach all goals while avoiding obstacles and other agents.
We implement:

- **Priority-Based Search (PBS)** for multi-agent planning on a grid  
- A **DQN** (deep RL) path planner as a low-level single-agent controller  
- An **RRT baseline** for comparison  
- 2D visualizations and log-based metrics for both approaches

## Repository Structure

Important files and folders:

- `environment.py`  
  Custom Gym-style grid environment used by PBS + DQN. Train with `train_model.py`.

- `final_model.zip`  
  Trained DQN policy. This is the model used in our experiments, run on 5 million timesteps.
  - Observation space: 6 observations of location in grid
  - Action space: wait, up, down, left, right
  - Reward function: Cost based on distance from goal (backward Dijkstra heuristic)
  - Constraints: Will either wait out or try and replan another action
  - Trained on a Gym environment (20x20 gridworld with grid obstacles) with Stable Baselines3 DQN


- `pbs.py`  
  Priority-Based Search multi-agent planner.  
  Can run in two main modes:
  - **PBS + DQN** (default, uses `final_model.zip`), and  
  - **PBS + RRT baseline** (with `--use_rrt`).

  Handles:
  - constructing the grid world,
  - spawning multiple agents with starts/goals,
  - running PBS to produce an ordering of priorities if collisions exist,
  - running the chosen low-level solver (DQN or RRT),
  - creating GIFs/PNGs of solutions

- `rrt_baseline.py`  
  Self-contained **single-agent RRT** demo and visualization on a grid.  
  Used as a baseline and for debugging the RRT implementation.

- `analyze_logs.py`  
  Script that reads the logs in `run_logs/` and produces aggregate metrics + plots:
  - success rate  
  - path length  
  - steps to goal, etc.

## Environment Setup

We recommend using a virtual environment with Python 3.9+.

git clone <repo-url>
cd 16362-final-project

python3 -m venv .venv
source .venv/bin/activate 

pip install --upgrade pip
pip install numpy matplotlib torch stable-baselines3 gym  

---

## Running the Code

**Run PBS with the trained DQN model:** 
`python3 pbs.py --model logs/final_model.zip --num_runs 1`

**Run PBS with the RRT baseline:**
`python3 pbs.py --use_rrt --num_runs 1`

**Analyze all logs in run_logs/ and generate aggregate plots:**
`python3 analyze_logs.py`

---

