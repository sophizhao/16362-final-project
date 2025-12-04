#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import yaml
import random
from pathlib import Path

import numpy as np
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from agents.environment import PathPlanningMaskEnv


'''
To train: python train_model.py --config config.yaml           
'''

def generate_random_obstacles(map_size=(10, 10), obstacle_density=0.1, seed=None):
    """
    Generate random obstacles in the grid.
    Ensures start (0,0) and goal (H-1, W-1) are free.
    """
    if seed is not None:
        np.random.seed(seed)
    
    H, W = map_size
    obstacles = []
    
    # Generate random obstacles
    for y in range(H):
        for x in range(W):
            # Skip start and goal positions
            if (y == 0 and x == 0) or (y == H-1 and x == W-1):
                continue
            
            if np.random.random() < obstacle_density:
                obstacles.append((y, x))
    
    return obstacles


class FlattenDictWrapper(gym.Wrapper):
    """
    Wrapper to flatten the dict observation space.
    Concatenates: scan, goal_vec, dist_grad, dist_phi, hist into single vector.
    """
    def __init__(self, env):
        super().__init__(env)
        obs_size = (
            env.observation_space['scan'].shape[0] +
            env.observation_space['goal_vec'].shape[0] +
            env.observation_space['dist_grad'].shape[0] +
            env.observation_space['dist_phi'].shape[0] +
            env.observation_space['hist'].shape[0]
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
    
    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs_dict), info
    
    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs_dict), reward, terminated, truncated, info
    
    def _flatten_obs(self, obs_dict):
        """Flatten observation dictionary into single vector."""
        return np.concatenate([
            obs_dict['scan'],
            obs_dict['goal_vec'],
            obs_dict['dist_grad'],
            obs_dict['dist_phi'],
            obs_dict['hist'].astype(np.float32)
        ], dtype=np.float32)


def make_env(map_size=(10, 10), obstacle_density=0.2, max_steps=200, hist_len=6, seed=None, rank=0):
    """
    Create a single environment instance.
    Each reset will generate new random obstacles.
    """
    def _init():
        # Set seed for this environment
        env_seed = seed + rank if seed is not None else None
        
        # Generate random obstacles
        obstacles = generate_random_obstacles(
            map_size=map_size,
            obstacle_density=obstacle_density,
            seed=env_seed
        )
        
        # Create environment
        env = PathPlanningMaskEnv(
            map_size=map_size,
            obstacles=obstacles,
            max_steps=max_steps,
            hist_len=hist_len
        )
        
        # Flatten dict observation
        env = FlattenDictWrapper(env)
        
        # Set seed
        if env_seed is not None:
            env.reset(seed=env_seed)
        
        return env
    
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--logdir", default="logs", help="Directory for logs and checkpoints")
    args = parser.parse_args()

    # Load config
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    # Environment settings
    seed              = int(cfg.get("seed", 42))
    map_size          = tuple(cfg.get("map_size", [10, 10]))
    obstacle_density  = float(cfg.get("obstacle_density", 0.2))
    max_steps         = int(cfg.get("max_steps", 200))
    hist_len          = int(cfg.get("hist_len", 6))
    n_envs            = int(cfg.get("n_envs", 1))

    # Training settings
    total_timesteps   = int(cfg.get("total_timesteps", 500_000))
    checkpoint_freq   = int(cfg.get("checkpoint_freq", 25_000))
    log_interval      = int(cfg.get("log_interval", 4))

    # DQN hyperparameters
    policy            = cfg.get("policy", "MlpPolicy")
    policy_kwargs     = cfg.get("policy_kwargs", {})
    learning_rate     = float(cfg.get("learning_rate", 1e-4))
    gamma             = float(cfg.get("gamma", 0.99))
    buffer_size       = int(cfg.get("buffer_size", 50_000))
    batch_size        = int(cfg.get("batch_size", 128))
    train_freq        = int(cfg.get("train_freq", 4))
    target_update     = int(cfg.get("target_update_interval", 1000))
    exploration_frac  = float(cfg.get("exploration_fraction", 0.3))
    exploration_final = float(cfg.get("exploration_final_eps", 0.05))
    learning_starts   = int(cfg.get("learning_starts", 1000))

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)

    # Create directories
    logdir         = Path(args.logdir)
    tb_dir         = logdir / "tensorboard"
    checkpoint_dir = logdir / "checkpoints"
    best_dir       = logdir / "best_model"
    
    for d in (logdir, tb_dir, checkpoint_dir, best_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Create vectorized environments
    env_fns = [
        make_env(
            map_size=map_size,
            obstacle_density=obstacle_density,
            max_steps=max_steps,
            hist_len=hist_len,
            seed=seed,
            rank=i
        )
        for i in range(n_envs)
    ]
    
    env = DummyVecEnv(env_fns)
    env = VecMonitor(env)

    # Create evaluation environment
    eval_env_fns = [
        make_env(
            map_size=map_size,
            obstacle_density=obstacle_density,
            max_steps=max_steps,
            hist_len=hist_len,
            seed=seed + 10000,  # Different seed for evaluation
            rank=0
        )
    ]
    eval_env = DummyVecEnv(eval_env_fns)
    eval_env = VecMonitor(eval_env)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq // n_envs,  # Adjust for number of environments
        save_path=str(checkpoint_dir),
        name_prefix="dqn_ckpt",
        save_vecnormalize=False
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(logdir),
        eval_freq=10000 // n_envs,  # Evaluate less frequently
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )

    # Build DQN model
    print(f"Training DQN on {map_size[0]}x{map_size[1]} grid")
    print(f"Obstacle density: {obstacle_density}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Number of environments: {n_envs}")
    print("\nNote: Using regular DQN with penalty for invalid actions")
    
    model = DQN(
        policy=policy,
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=train_freq,
        target_update_interval=target_update,
        exploration_fraction=exploration_frac,
        exploration_final_eps=exploration_final,
        learning_starts=learning_starts,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tb_dir),
        verbose=1,
        seed=seed
    )

    # Train
    print("\nStarting training...")
    print(f"Learning will start after {learning_starts} steps")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_cb,  # Only checkpoint for now
            progress_bar=True,
            log_interval=log_interval
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing environments...")
        env.close()
        eval_env.close()

    # Save final model
    final_model_path = logdir / "final_model"
    model.save(str(final_model_path))
    print(f"\nTraining complete. Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()