#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import yaml
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from environment import PathPlanningMaskEnv


class EpisodeLogger(BaseCallback):
    """Print episode rewards/lengths whenever VecMonitor records one."""

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [])
        for entry in info:
            if "episode" in entry:
                ep = entry["episode"]
                print(
                    f"[Episode] reward={ep['r']:.2f} length={int(ep['l'])} steps"
                )
        return True

'''
python train_model.py \
  --config configs/ppo_grid.yaml \
  --logdir results/grid_experiment
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to ppo_config.yaml")
    parser.add_argument("--logdir", default="logs", help="Output directory")
    args = parser.parse_args()

    cfg           = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed          = int(cfg.get("seed", 0))
    env_kwargs    = dict(cfg.get("env_kwargs", {}))

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

    n_steps        = int(cfg.get("n_steps",    2048))
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
    def make_train_env(rank: int):
        def _init():
            env = PathPlanningMaskEnv(seed=seed + rank, **env_kwargs)
            env.time_cost = time_cost
            env.completion_bonus = completion_bonus
            return env
        return _init

    if n_envs > 1:
        train_env = SubprocVecEnv(
            [make_train_env(i) for i in range(n_envs)],
            start_method="spawn"
        )
    else:
        train_env = DummyVecEnv([make_train_env(0)])
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(
        train_env,
        norm_obs=True, norm_reward=False,
        norm_obs_keys=['scan', 'goal_vec', 'dist_grad', 'dist_phi'],
        clip_obs=10.0
    )

    # callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(best_dir),
        name_prefix="ppo_ckpt"
    )
    log_cb = EpisodeLogger()

    # Train
    model = PPO(
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
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, log_cb])
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
