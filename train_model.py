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

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO

from domains_cc.map_and_scen_utils import parse_scen_file
from domains_cc.path_planning_rl.env.planning_env import PathPlanningMaskEnv

# ─── RandomResetEnv: Randomly select scenario on each reset ─────────────────────
class RandomResetEnv(gym.Env):
    def __init__(self,
                 scen_files,
                 dynamics_config,
                 footprint_config,
                 time_cost=None,
                 completion_bonus=None,
                 seed=None,
                 **env_kwargs):
        super().__init__()
        self.scen_files        = scen_files
        self.dynamics_config   = dynamics_config
        self.footprint_config  = footprint_config
        self._time_cost        = time_cost
        self._completion_bonus = completion_bonus
        self.env_kwargs        = env_kwargs

        if seed is not None:
            random.seed(seed)

        self._make_env()
        self.observation_space = self.env.observation_space
        self.action_space      = self.env.action_space

    def _make_env(self):
        scen = random.choice(self.scen_files)
        pairs = parse_scen_file(scen)
        idx   = random.randrange(len(pairs))
        base = PathPlanningMaskEnv(
            scen_file        = scen,
            problem_index    = idx,
            dynamics_config  = self.dynamics_config,
            footprint_config = self.footprint_config,
            **self.env_kwargs
        )
        if self._time_cost is not None:
            base.time_cost = self._time_cost
        if self._completion_bonus is not None:
            base.completion_bonus = self._completion_bonus
        self.env = base

    # def reset(self, **kwargs):
    #     # 0.5 chance to switch problem
    #     switch_problem = random.random() < 0.8

    #     if switch_problem or not hasattr(self, "env"):
    #         # print("Switched")
    #         # full reset with a new scenario/problem pair
    #         self._make_env()
    #         return self.env.reset(**kwargs)
    #     else:
    #         # print("Kept")
    #         # keep the same environment, force a new constraint
    #         return self.env.reset(force_constraint=True, **kwargs)

    def reset(self, **kwargs):
        self._make_env()
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def action_masks(self):
        return self.env.action_masks()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
'''
python -m domains_cc.path_planning_rl.train.train_ppo \
  --config domains_cc/path_planning_rl/configs/ppo_config.yaml \
  --logdir results/ppo_rectangle_unicycle
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to ppo_config.yaml")
    parser.add_argument("--logdir", default="logs", help="Output directory")
    args = parser.parse_args()

    cfg              = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed             = int(cfg.get("seed", 0))
    scen_files       = cfg["scen_files"]
    dynamics_config  = cfg["dynamics_config"]
    footprint_config = cfg["footprint_config"]

    total_timesteps  = int(cfg.get("total_timesteps", 2_000_000))
    n_envs           = int(cfg.get("n_envs", 8))
    checkpoint_freq  = int(cfg.get("checkpoint_freq", cfg.get("eval_freq", 200_000)))

    time_cost        = float(cfg.get("time_cost",       -0.1))
    completion_bonus = float(cfg.get("completion_bonus",  50.0))

    policy           = cfg.get("policy", "MultiInputPolicy")
    policy_kwargs    = cfg.get("policy_kwargs", {})
    lr_cfg           = cfg.get("learning_rate", "linear")
    if isinstance(lr_cfg, str) and lr_cfg.lower() == "linear":
        init_lr       = float(cfg.get("init_learning_rate", 3e-4))
        learning_rate = lambda p: init_lr * p
    else:
        learning_rate = float(lr_cfg)

    n_steps        = int(cfg.get("n_steps",    4096))
    batch_size     = int(cfg.get("batch_size", 512))
    n_epochs       = int(cfg.get("n_epochs",   10))
    gamma          = float(cfg.get("gamma",    0.99))
    clip_range     = float(cfg.get("clip_range",0.2))
    ent_coef       = float(cfg.get("ent_coef", 0.01))
    max_grad_norm  = float(cfg.get("max_grad_norm",0.5))

    logdir   = Path(args.logdir)
    tb_dir   = logdir / "tensorboard"
    best_dir = logdir / "best_model"
    for d in (logdir, tb_dir, best_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Training env factory
    def make_train_env(rank):
        def _init():
            return ActionMasker(
                RandomResetEnv(
                    scen_files, dynamics_config, footprint_config,
                    time_cost=time_cost,
                    completion_bonus=completion_bonus,
                    seed=seed + rank
                ),
                lambda e: e.action_masks()
            )
        return _init

    train_env = SubprocVecEnv(
        [make_train_env(i) for i in range(n_envs)],
        start_method="fork"
    )
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(
        train_env,
        norm_obs=True, norm_reward=False,
        norm_obs_keys=['scan', 'goal_vec', 'dist_grad', 'dist_phi'],
        clip_obs=10.0
    )

    # Only keep checkpointing
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(best_dir),
        name_prefix="ppo_ckpt"
    )

    # Train
    model = MaskablePPO(
        policy=policy,
        env=train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        clip_range=clip_range,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        seed=seed,
        tensorboard_log=str(tb_dir),
        device="auto",
        verbose=1,
    )
    try:
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_cb)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        train_env.close()

    # Save final artifacts
    model.save(str(logdir / "final_model_no_constraints_new_retrain"))
    train_env.save(str(logdir / "vecnormalize_no_constraints_retrain.pkl"))
    print("Training complete. Outputs in", args.logdir)


if __name__ == "__main__":
    main()