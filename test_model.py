import argparse
import numpy as np
from stable_baselines3 import DQN
from environment import PathPlanningMaskEnv
import time
import matplotlib.pyplot as plt

'''
To test: python test_model.py --model logs/final_model --episodes 5
'''

def generate_random_obstacles(map_size=(10, 10), obstacle_density=0.2, seed=None):
    """Generate random obstacles in the grid."""
    if seed is not None:
        np.random.seed(seed)
    
    H, W = map_size
    obstacles = []
    
    for y in range(H):
        for x in range(W):
            if (y == 0 and x == 0) or (y == H-1 and x == W-1):
                continue
            
            if np.random.random() < obstacle_density:
                obstacles.append((y, x))
    
    return obstacles


class FlattenDictWrapper:
    """Wrapper to flatten dict observations for the trained model."""
    def __init__(self, env):
        self.env = env
    
    def flatten_obs(self, obs_dict):
        """Flatten observation dictionary into single vector."""
        return np.concatenate([
            obs_dict['scan'],
            obs_dict['goal_vec'],
            obs_dict['dist_grad'],
            obs_dict['dist_phi'],
            obs_dict['hist'].astype(np.float32)
        ], dtype=np.float32)


def test_model(model_path, num_episodes=10, obstacle_density=0.2, 
               render=True, deterministic=True, seed=None, animate=False, delay=0.1):
    """
    Test a trained model on random environments.
    
    Args:
        model_path: Path to saved model (.zip file)
        num_episodes: Number of test episodes
        obstacle_density: Density of obstacles (0.0 to 1.0)
        render: Whether to render the environment
        deterministic: Use deterministic policy
        seed: Random seed for reproducibility
        animate: Show step-by-step animation
        delay: Delay between animation steps (seconds)
    """
    print(f"Loading model from: {model_path}")
    model = DQN.load(model_path)
    print("Model loaded successfully!\n")
    
    success_count = 0
    collision_count = 0
    timeout_count = 0
    total_rewards = []
    episode_lengths = []
    
    wrapper = FlattenDictWrapper(None)
    
    for episode in range(num_episodes):
        print("=" * 60)
        print(f"Episode {episode + 1}/{num_episodes}")
        print("=" * 60)
        
        # Generate new random environment
        episode_seed = seed + episode if seed is not None else None
        obstacles = generate_random_obstacles(
            map_size=(10, 10),
            obstacle_density=obstacle_density,
            seed=episode_seed
        )
        
        env = PathPlanningMaskEnv(
            map_size=(10, 10),
            obstacles=obstacles,
            max_steps=200,
            hist_len=6
        )
        
        wrapper.env = env
        

        obs_dict, info = env.reset(seed=episode_seed)
        obs_flat = wrapper.flatten_obs(obs_dict)
        
        print(f"Start: {env.agent_pos}, Goal: {env.goal}")
        print(f"Number of obstacles: {len(obstacles)}")
        
        if render:
            print("\nInitial state:")
            env.render()
            if animate:
                time.sleep(delay * 3) 
        
        episode_reward = 0
        steps = 0
        done = False
        
        action_names = ['stay', 'up', 'down', 'left', 'right']
        
        while not done:
            # Predict action
            action, _states = model.predict(obs_flat, deterministic=deterministic)
            action_int = int(action)
            
            # Take action
            obs_dict, reward, terminated, truncated, info = env.step(action_int)
            obs_flat = wrapper.flatten_obs(obs_dict)
            
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            if animate:
                print(f"  Step {steps}: {action_names[action_int]} -> {env.agent_pos}, reward={reward:.2f}")
            
            if render and animate:
                env.render()
                plt.pause(delay)  
        
        if render:
            if animate:
                print(f"\nFinal state:")
            env.render()
            plt.pause(2.0) 
            
        if render and hasattr(env, '_fig') and env._fig is not None:
            plt.close(env._fig)
            env._fig = None
        
        # Record statistics
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        if info['reached']:
            success_count += 1
            print(f"✓ SUCCESS! Reached goal in {steps} steps")
        elif info['collision']:
            collision_count += 1
            print(f"✗ COLLISION at step {steps}")
        else:
            timeout_count += 1
            print(f"✗ TIMEOUT after {steps} steps")
        
        print(f"Episode reward: {episode_reward:.2f}")
        print()
    
    # Print summary statistics
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total episodes: {num_episodes}")
    print(f"Success rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"Collision rate: {collision_count}/{num_episodes} ({100*collision_count/num_episodes:.1f}%)")
    print(f"Timeout rate: {timeout_count}/{num_episodes} ({100*timeout_count/num_episodes:.1f}%)")
    print(f"\nAverage reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    if success_count > 0:
        successful_lengths = [episode_lengths[i] for i in range(len(total_rewards)) 
                             if total_rewards[i] > 0]  # Approximate successful episodes
        if successful_lengths:
            print(f"Average steps to goal (successful): {np.mean(successful_lengths):.1f}")
    
    print("=" * 60)
    
    return {
        'success_rate': success_count / num_episodes,
        'collision_rate': collision_count / num_episodes,
        'timeout_rate': timeout_count / num_episodes,
        'avg_reward': np.mean(total_rewards),
        'avg_length': np.mean(episode_lengths),
        'total_rewards': total_rewards,
        'episode_lengths': episode_lengths
    }


def test_single_environment(model_path, visualize_policy=False, animate=True, delay=0.3):
    """
    Test model on a single environment with detailed visualization.
    
    Args:
        model_path: Path to saved model
        visualize_policy: Show Q-values at each step
        animate: Show step-by-step animation
        delay: Delay between steps
    """
    print("=" * 60)
    print("SINGLE ENVIRONMENT TEST")
    print("=" * 60)
    
    # Load model
    model = DQN.load(model_path)
    
    # Create specific test environment
    obstacles = [(2, 5), (3, 5), (4, 5), (5, 5), (6, 5)]
    env = PathPlanningMaskEnv(
        map_size=(10, 10),
        obstacles=obstacles,
        max_steps=100,
        hist_len=6
    )
    
    wrapper = FlattenDictWrapper(env)
    
    obs_dict, _ = env.reset(seed=42)
    obs_flat = wrapper.flatten_obs(obs_dict)
    
    print(f"Start: {env.agent_pos}, Goal: {env.goal}")
    print("\nInitial state:")
    env.render()
    plt.pause(1.0)
    
    step = 0
    done = False
    total_reward = 0
    
    print("\nStep-by-step execution:")
    print("-" * 60)
    
    action_names = ['stay', 'up', 'down', 'left', 'right']
    
    while not done and step < 50:
        if visualize_policy:
            q_values = model.q_net(model.policy.obs_to_tensor(obs_flat)[0])
            q_values = q_values.detach().cpu().numpy()[0]
            print(f"\nStep {step}: Q-values = {q_values}")
        
        action, _ = model.predict(obs_flat, deterministic=True)
        action_int = int(action)
        
        obs_dict, reward, terminated, truncated, info = env.step(action_int)
        obs_flat = wrapper.flatten_obs(obs_dict)
        
        total_reward += reward
        step += 1
        done = terminated or truncated
        
        print(f"Step {step}: Action={action_names[action_int]} ({action_int}), "
              f"Reward={reward:.2f}, Position={env.agent_pos}")
        
        if animate:
            env.render()
            plt.pause(delay)
        
        if done:
            if not animate:
                env.render()
            print(f"\nFinal state:")
            env.render()
            if info['reached']:
                print(f"✓ Reached goal in {step} steps!")
            else:
                print(f"✗ Failed after {step} steps")
            plt.pause(3.0) 
    
    print(f"\nTotal reward: {total_reward:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Test trained DQN model')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model (e.g., logs/final_model.zip)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of test episodes')
    parser.add_argument('--obstacle_density', type=float, default=0.2,
                       help='Obstacle density (0.0 to 1.0)')
    parser.add_argument('--no_render', action='store_true',
                       help='Disable rendering')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy instead of deterministic')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--single', action='store_true',
                       help='Run single environment test with detailed output')
    parser.add_argument('--visualize_policy', action='store_true',
                       help='Show Q-values at each step (only with --single)')
    parser.add_argument('--animate', action='store_true',
                       help='Show step-by-step animation')
    parser.add_argument('--delay', type=float, default=0.2,
                       help='Delay between animation steps in seconds')
    
    args = parser.parse_args()
    
    if args.single:
        test_single_environment(args.model, args.visualize_policy, args.animate, args.delay)
    else:
        results = test_model(
            model_path=args.model,
            num_episodes=args.episodes,
            obstacle_density=args.obstacle_density,
            render=not args.no_render,
            deterministic=not args.stochastic,
            seed=args.seed,
            animate=args.animate,
            delay=args.delay
        )


if __name__ == "__main__":
    main()