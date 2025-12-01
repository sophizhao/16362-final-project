# maskeddqn_wrapper.py
import torch
import numpy as np
from stable_baselines3 import DQN


class MaskedDQNWrapper(DQN):
    """
    A lightweight wrapper that applies action masks to the Q-values
    at inference time. Works with any stable_baselines3 DQN model.
    """

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        # Get the raw action from DQN
        # but first extract the mask
        if isinstance(observation, dict):
            obs = observation["obs"]
            mask = observation["action_mask"]
        else:
            raise ValueError("Environment must return {'obs': ..., 'action_mask': ...}")

        # Convert to batch form because SB3 always expects [batch, ...]
        obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(self.device)

        # Get Q-values from the underlying DQN policy
        q_values = self.q_net(obs_tensor)
        q_values = q_values.detach().cpu().numpy()[0]

        # Apply mask → masked actions get -inf so argmax ignores them
        masked_q = q_values.copy()
        masked_q[mask == 0] = -np.inf

        # Choose best valid action
        action = int(np.argmax(masked_q))

        return action, state
