#!/usr/bin/env python3
"""
Priority-Based Search (PBS) implementation for multi-agent path planning.
Aligns with Jiaoyang-Li/PBS architecture using DQN agents as low-level planners.
"""

import argparse
import numpy as np
from stable_baselines3 import DQN
from environment import PathPlanningMaskEnv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from collections import deque, defaultdict
import copy
import time
import heapq

class Agent:
    """
    Represents a single agent. 
    Acts as the 'Low Level Planner' wrapping the learned policy.
    """
    def __init__(self, agent_id, model, start_pos, goal_pos, color, map_size, static_obstacles, max_steps):
        self.id = agent_id
        self.model = model
        self.start = np.array(start_pos)
        self.goal = np.array(goal_pos)
        self.color = color
        self.map_size = map_size
        self.static_obstacles = static_obstacles
        self.max_steps = max_steps
        self.path = []  # List of (y, x, t)

    def find_path(self, dynamic_obstacles):
        """
        Plan a path avoiding static obstacles and higher-priority agent paths (dynamic_obstacles).
        dynamic_obstacles: list of (y, x, t) occupied by higher priority agents.
        """
        # Initialize environment for this specific search
        env = PathPlanningMaskEnv(
            map_size=self.map_size,
            obstacles=self.static_obstacles,
            max_steps=self.max_steps,
            hist_len=6
        )
        
        env.agent_pos = self.start.copy()
        env.goal = self.goal.copy()
        
        # Precompute distance field (Dijkstra/BFS style heuristic for the Env)
        env.dist_map = env._compute_distance_map(env.goal)
        reachable = np.isfinite(env.dist_map)
        env.max_potential = float(np.max(env.dist_map[reachable])) if np.any(reachable) else 1.0
        env.dist_map[~reachable] = env.max_potential
        env._compute_gradient_field()
        
        # Reset observation
        env.step_count = 0
        env.action_history.clear()
        for _ in range(env.hist_len):
            env.action_history.append(0)
            
        wrapper = FlattenDictWrapper(env)
        obs_dict = env._compute_obs()
        obs_flat = wrapper.flatten_obs(obs_dict)
        
        path = []
        timestep = 0
        
        # Convert dynamic obstacles list to a set for O(1) lookup: {(y,x,t)}
        constraint_set = set(dynamic_obstacles)

        for step in range(self.max_steps):
            current_pos = (env.agent_pos[0], env.agent_pos[1], timestep)
            path.append(current_pos)
            
            if np.array_equal(env.agent_pos, self.goal):
                break
            
            # 1. Get preferred action from DQN
            action, _ = self.model.predict(obs_flat, deterministic=True)
            action_int = int(action)
            
            # 2. Check validity against Dynamic Obstacles (Higher Priority Agents)
            moves = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
            dy, dx = moves[action_int]
            next_y, next_x = env.agent_pos[0] + dy, env.agent_pos[1] + dx
            next_state = (next_y, next_x, timestep + 1)
            
            # If preferred action hits a higher priority agent, wait (action 0)
            # If waiting also hits them (vertex conflict), we are stuck (simple collision logic)
            if next_state in constraint_set:
                action_int = 0 # Try to wait
                next_state_wait = (env.agent_pos[0], env.agent_pos[1], timestep + 1)
                if next_state_wait in constraint_set:
                    # Even waiting causes collision. In a full search, we might backtrack.
                    # Here, with DQN, we just take the collision or stay put.
                    pass 

            # 3. Step Environment
            obs_dict, reward, terminated, truncated, info = env.step(action_int)
            obs_flat = wrapper.flatten_obs(obs_dict)
            timestep += 1
            
            if terminated or truncated:
                break
                
        # Fill remaining steps if reached goal early to ensure safety for lower priority agents
        last_pos = path[-1]
        for t in range(len(path), self.max_steps + 5): # Buffer
            path.append((last_pos[0], last_pos[1], t))
          
        print(path)
        return path

class PBSNode:
    """
    Node in the PBS High-Level Search Tree.
    Contains partial ordering constraints and the current solution plan.
    """
    def __init__(self, priorities=None, solution=None, cost=0):
        # priorities: Set of tuples (high_id, low_id) -> high_id has priority over low_id
        self.priorities = set(priorities) if priorities else set()
        # solution: Dict {agent_id: path}
        self.solution = solution if solution else {}
        self.cost = cost
        self.conflicts = [] 
    
    def __lt__(self, other):
        # For PriorityQueue: lower cost is better, then fewer conflicts
        return self.cost < other.cost

    def get_priority_graph(self, num_agents):
        """Returns adjacency list for topological sort."""
        adj = defaultdict(list)
        in_degree = {i: 0 for i in range(num_agents)}
        for high, low in self.priorities:
            adj[high].append(low)
            in_degree[low] += 1
        return adj, in_degree

class FlattenDictWrapper:
    """Wrapper to flatten dict observations for Stable Baselines3."""
    def __init__(self, env):
        self.env = env
    
    def flatten_obs(self, obs_dict):
        return np.concatenate([
            obs_dict['scan'],
            obs_dict['goal_vec'],
            obs_dict['dist_grad'],
            obs_dict['dist_phi'],
            obs_dict['hist'].astype(np.float32)
        ], dtype=np.float32)

class PBSPlanner:
    """
    Priority-Based Search (PBS) Main Solver.
    """
    def __init__(self, map_size=(10, 10), obstacles=None, max_steps=200):
        self.map_size = map_size
        self.obstacles = obstacles if obstacles else []
        self.max_steps = max_steps
        self.agents = {} # id -> Agent object
        self.agent_ids = []
        
        # Stats
        self.generated_nodes = 0
        self.expanded_nodes = 0

    def add_agent(self, agent_id, model_path, start_pos, goal_pos, color):
        model = DQN.load(model_path)
        agent = Agent(agent_id, model, start_pos, goal_pos, color, 
                      self.map_size, self.obstacles, self.max_steps)
        self.agents[agent_id] = agent
        self.agent_ids.append(agent_id)

    def topological_sort(self, priorities):
        """
        Sort agents based on priority constraints.
        Returns: list of agent_ids in planning order, or None if cyclic.
        """
        adj = defaultdict(list)
        in_degree = {i: 0 for i in self.agent_ids}
        
        for high, low in priorities:
            adj[high].append(low)
            in_degree[low] += 1
            
        queue = deque([i for i in self.agent_ids if in_degree[i] == 0])
        order = []
        
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
                    
        if len(order) != len(self.agent_ids):
            return None # Cycle detected
        return order

    def update_plan(self, node, agents_to_replan):
        """
        Replans the path for specific agents given the node's priorities.
        In strict PBS, we only replan the agent that got a new lower-priority constraint.
        """
        # 1. Determine Global Planning Order based on priorities
        order = self.topological_sort(node.priorities)
        if order is None:
            return False # Invalid priority (cycle)

        # 2. Collect Space-Time Constraints (Occupied cells by higher priority agents)
        # We must iterate in topological order.
        occupied_spacetime = set()
        
        for agent_id in order:
            agent = self.agents[agent_id]
            
            # If agent needs replanning OR has no path yet
            if agent_id in agents_to_replan or agent_id not in node.solution:
                # Plan path considering all currently occupied space-time slots
                # Note: occupied_spacetime only contains paths of agents earlier in 'order'
                path = agent.find_path(list(occupied_spacetime))
                node.solution[agent_id] = path
            
            # Add this agent's path to constraints for subsequent agents
            current_path = node.solution[agent_id]
            for y, x, t in current_path:
                occupied_spacetime.add((y, x, t))
                
        # Calculate Cost
        node.cost = sum(len(path) for path in node.solution.values())
        return True

    def find_collisions(self, solution):
        """
        Find the first collision between any two agents.
        Returns: (agent_i, agent_j, location, time) or None
        """
        # Create a lookup: (y,x,t) -> agent_id
        occupied = {}
        
        # Check Vertex Collisions
        max_len = max(len(p) for p in solution.values())
        
        for t in range(max_len):
            pos_map = {}
            for aid, path in solution.items():
                if t < len(path):
                    pos = (path[t][0], path[t][1])
                    if pos in pos_map:
                        return (pos_map[pos], aid, pos, t)
                    pos_map[pos] = aid
        
        # Note: Edge collisions (swapping cells) are ignored for simplicity 
        # as per standard DQN grid implementations, but PBS usually checks them.
        return None

    def solve(self):
        print(f"Starting PBS for {len(self.agents)} agents...")
        
        # 1. Root Node
        root = PBSNode()
        # Initially, plan all agents (empty priorities -> order doesn't matter yet, 
        # but topological sort will give default order)
        success = self.update_plan(root, set(self.agent_ids))
        if not success: return None
        
        # 2. Check root conflicts
        collision = self.find_collisions(root.solution)
        if collision is None:
            return root.solution
            
        root.conflicts = [collision]
        
        # 3. Stack for DFS (Depth First Search is standard for PBS to save memory)
        # Use a PriorityQueue (Open List) if you want Best-First Search
        open_list = [root] 
        self.generated_nodes = 1
        
        while open_list:
            curr_node = open_list.pop() # DFS
            self.expanded_nodes += 1
            
            collision = self.find_collisions(curr_node.solution)
            
            if collision is None:
                print(f"Solution found! Cost: {curr_node.cost}")
                return curr_node.solution
            
            a1, a2, loc, t = collision
            print(f"Node {self.expanded_nodes}: Collision {a1}-{a2} at {loc} t={t}")
            
            # Branching: Resolve (a1, a2) collision
            # Option 1: a1 > a2 (a1 has priority)
            child1 = PBSNode(priorities=copy.deepcopy(curr_node.priorities), 
                             solution=copy.deepcopy(curr_node.solution))
            child1.priorities.add((a1, a2))
            
            # Only replan a2 (the lower priority one)
            if self.update_plan(child1, {a2}):
                open_list.append(child1)
                self.generated_nodes += 1
                
            # Option 2: a2 > a1 (a2 has priority)
            child2 = PBSNode(priorities=copy.deepcopy(curr_node.priorities), 
                             solution=copy.deepcopy(curr_node.solution))
            child2.priorities.add((a2, a1))
            
            # Only replan a1
            if self.update_plan(child2, {a1}):
                open_list.append(child2)
                self.generated_nodes += 1
                
        print("PBS failed to find a solution.")
        return None

    def visualize_solution(self, solution, animate=True, delay=0.1, save_path=None):
        """Visualizes the found solution."""
        if not solution:
            print("No solution to visualize.")
            return

        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Calculate max time
        max_time = max(max(t for _, _, t in path) for path in solution.values())
        
        if animate:
            plt.ion()
            
        for timestep in range(max_time + 1):
            ax.clear()
            
            # Draw Grid & Obstacles
            ax.set_xlim(0, self.map_size[1])
            ax.set_ylim(0, self.map_size[0])
            ax.set_xticks(range(self.map_size[1] + 1))
            ax.set_yticks(range(self.map_size[0] + 1))
            ax.grid(True, linestyle='--', alpha=0.5)
            
            for (y, x) in self.obstacles:
                rect = Rectangle((x, y), 1, 1, facecolor='black', alpha=0.6)
                ax.add_patch(rect)
                
            # Draw Agents
            for agent_id, path in solution.items():
                agent = self.agents[agent_id]
                
                # Goal
                ax.add_patch(Circle((agent.goal[1]+0.5, agent.goal[0]+0.5), 0.2, 
                                  color=agent.color, alpha=0.3))
                
                # Current Position
                if timestep < len(path):
                    curr_y, curr_x, _ = path[timestep]
                    ax.add_patch(Circle((curr_x+0.5, curr_y+0.5), 0.35, 
                                      color=agent.color, label=f'A{agent_id}'))
                    ax.text(curr_x+0.5, curr_y+0.5, str(agent_id), 
                           color='white', ha='center', va='center', weight='bold')
                
            ax.set_title(f"PBS Solution - Time: {timestep}/{max_time}")
            ax.invert_yaxis() # Matrix coordinates
            
            if animate:
                plt.draw()
                plt.pause(delay)
            elif timestep == 0:
                plt.show() # Just show start if not animating
                
        if save_path:
            plt.savefig(save_path)
            
        if animate:
            plt.ioff()
            plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to DQN model')
    parser.add_argument('--num_agents', type=int, default=4)
    parser.add_argument('--map_size', type=int, nargs=2, default=[10, 10])
    parser.add_argument('--obstacle_density', type=float, default=0.1)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_anim', action='store_true')
    args = parser.parse_args()
    
    # Generate Map
    H, W = args.map_size
    obstacles = []
    for y in range(H):
        for x in range(W):
            if np.random.random() < args.obstacle_density:
                obstacles.append((y, x))
                
    planner = PBSPlanner(map_size=(H, W), obstacles=obstacles, max_steps=args.max_steps)
    
   # 1. Identify all free cells
    free_cells = [(y, x) for y in range(H) for x in range(W) if (y, x) not in obstacles]
    num_free = len(free_cells)

    if num_free < args.num_agents:
        raise ValueError("Map is too small / too many obstacles for this number of agents.")

    # 2. Select Unique Start Positions
    # We pick N unique indices from the free_cells list
    start_indices = np.random.choice(num_free, args.num_agents, replace=False)
    starts = [free_cells[i] for i in start_indices]

    # 3. Select Unique Goal Positions
    goals = []
    chosen_goal_set = set()

    for i in range(args.num_agents):
        while True:
            # Pick a random candidate for the goal
            idx = np.random.randint(num_free)
            candidate = free_cells[idx]

            # Constraint A: Goal must be unique (no two agents have same goal)
            # Constraint B: Goal cannot be the agent's own start position
            if (candidate not in chosen_goal_set) and (candidate != starts[i]):
                goals.append(candidate)
                chosen_goal_set.add(candidate)
                break

    # 4. Add Agents to Planner
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'brown', 'pink']
    
    print("Map Setup:")
    for i in range(args.num_agents):
        start_pos = starts[i]
        goal_pos = goals[i]
        color = colors[i % len(colors)]
        
        planner.add_agent(i, args.model, start_pos, goal_pos, color)
        print(f"  Agent {i}: {start_pos} -> {goal_pos}")
    # Solve
    start_time = time.time()
    solution = planner.solve()
    duration = time.time() - start_time
    
    if solution:
        print(f"\nSolved in {duration:.2f}s")
        print(f"Nodes Generated: {planner.generated_nodes}")
        print(f"Nodes Expanded: {planner.expanded_nodes}")
        planner.visualize_solution(solution, animate=not args.no_anim)
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()