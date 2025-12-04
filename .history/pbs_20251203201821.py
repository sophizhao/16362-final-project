#!/usr/bin/env python3
"""
Priority-Based Search (PBS) implementation for multi-agent path planning.
Aligns with Jiaoyang-Li/PBS architecture using DQN agents as low-level planners.
"""

import argparse
import os
import numpy as np
from stable_baselines3 import DQN
from environment import PathPlanningMaskEnv
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from collections import deque, defaultdict
import copy
import time
import heapq
from rrt_baseline import rrt_plan

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
        self.path = []  # List of (y, x, timestamp)

    def find_path(self, vertex_constraints, edge_constraints=None):
        """
        Plan a path avoiding static obstacles and higher-priority agent paths.
        
        vertex_constraints: list of (y, x, t) occupied by higher priority agents.
        edge_constraints: set of (from_y, from_x, to_y, to_x, t) representing forbidden moves
                         (because another agent is moving in the opposite direction).
        """
        edge_constraints = edge_constraints or set()
        env = PathPlanningMaskEnv(
            map_size=self.map_size,
            obstacles=self.static_obstacles,
            max_steps=self.max_steps,
            hist_len=6
        )
        
        env.agent_pos = self.start.copy()
        env.goal = self.goal.copy()
        

        # backward dijkstra heuristic 
        env.dist_map = env._compute_distance_map(env.goal)
        reachable = np.isfinite(env.dist_map)
        env.max_potential = float(np.max(env.dist_map[reachable])) if np.any(reachable) else 1.0
        env.dist_map[~reachable] = env.max_potential
        env._compute_gradient_field()
        
        env.step_count = 0
        env.action_history.clear()
        for _ in range(env.hist_len):
            env.action_history.append(0)
            
        wrapper = FlattenDictWrapper(env)
        obs_dict = env._compute_obs()
        obs_flat = wrapper.flatten_obs(obs_dict)
        
        timestep = 0
        path = [(env.agent_pos[0], env.agent_pos[1], timestep)]

        # O(1) lookup
        vertex_set = set(vertex_constraints)
        edge_set = set(edge_constraints)

        for step in range(self.max_steps):
            if np.array_equal(env.agent_pos, self.goal):
                break
            
            # get predicted action
            action, _ = self.model.predict(obs_flat, deterministic=True)
            action_int = int(action)
            
            # next step for agent
            moves = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
            curr_y, curr_x = env.agent_pos
            dy, dx = moves[action_int]
            next_y, next_x = curr_y + dy, curr_x + dx
            next_t = timestep + 1
            
            # vertex constraint, if its occupied rn
            vertex_blocked = (next_y, next_x, next_t) in vertex_set
            
            # edge constraint, swapping
            edge_blocked = (curr_y, curr_x, next_y, next_x, timestep) in edge_set
            
            if vertex_blocked or edge_blocked:
                # can't really inject actions with DQN (not maskable with API)
                # just wait
                wait_pos = (curr_y, curr_x, next_t)
                if wait_pos not in vertex_set:
                    action_int = 0  # wait
                    block_reason = "VERTEX" if vertex_blocked else "EDGE"
                    print(f"  [PBS] Agent {self.id} t={timestep}: forced WAIT at ({curr_y},{curr_x}) due to {block_reason} constraint (wanted ({next_y},{next_x}))")
                else:
                    # replan?!
                    found_valid = False
                    for alt_action in [1, 2, 3, 4]:  # up, down, left, right
                        if alt_action == action_int:
                            continue
                        ady, adx = moves[alt_action]
                        alt_y, alt_x = curr_y + ady, curr_x + adx
                        alt_vertex_ok = (alt_y, alt_x, next_t) not in vertex_set
                        alt_edge_ok = (curr_y, curr_x, alt_y, alt_x, timestep) not in edge_set
                        in_bounds = 0 <= alt_y < self.map_size[0] and 0 <= alt_x < self.map_size[1]
                        not_static = (alt_y, alt_x) not in self.static_obstacles
                        if alt_vertex_ok and alt_edge_ok and in_bounds and not_static:
                            action_int = alt_action
                            found_valid = True
                            action_names = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
                            print(f"  [PBS] Agent {self.id} t={timestep}: rerouted to ({alt_y},{alt_x}) [{action_names[alt_action]}] (wanted ({next_y},{next_x}))")
                            break
                    if not found_valid:
                        action_int = 0  # Forced wait, may cause issues
                        print(f"  [PBS] Agent {self.id} t={timestep}: FORCED WAIT at ({curr_y},{curr_x}) - no valid alternatives!") 


            obs_dict, reward, terminated, truncated, info = env.step(action_int)
            obs_flat = wrapper.flatten_obs(obs_dict)
            timestep += 1
            path.append((env.agent_pos[0], env.agent_pos[1], timestep))
            
            # found goal
            if terminated or truncated:
                break
        last_pos = path[-1]
        for t in range(len(path), self.max_steps + 5):
            path.append((last_pos[0], last_pos[1], t))
          
        # print(path)
        return path

class RRTAgent:
    """
    Lightweight agent that uses the single-agent RRT baseline to compute a path.
    Mimics the PBS.Agent API so it can be swapped in as a different low-level planner.
    """
    def __init__(
        self,
        agent_id,
        start_pos,
        goal_pos,
        color,
        map_size,
        static_obstacles,
        max_steps,
        rrt_max_iters=4000,
        rrt_step_size=1,
        rrt_goal_bias=0.1,
        rrt_max_retries=10,
    ):
        self.id = agent_id
        self.start = np.array(start_pos)
        self.goal = np.array(goal_pos)
        self.color = color
        self.map_size = map_size
        self.static_obstacles = static_obstacles or []
        self.max_steps = max_steps
        self.rrt_max_iters = rrt_max_iters
        self.rrt_step_size = rrt_step_size
        self.rrt_goal_bias = rrt_goal_bias
        self.rrt_max_retries = rrt_max_retries
        self.base_grid = self._build_occupancy_grid()
        self.path = []

    def _build_occupancy_grid(self):
        grid = np.zeros(self.map_size, dtype=int)
        for (y, x) in self.static_obstacles:
            if 0 <= y < self.map_size[0] and 0 <= x < self.map_size[1]:
                grid[y, x] = 1
        return grid

    def _densify_path(self, coords):
        """Expand straight-line RRT segments into 4-connected moves respecting static obstacles."""
        if len(coords) <= 1:
            return coords
        dense = [coords[0]]
        max_y, max_x = self.map_size
        for i in range(len(coords) - 1):
            target_y, target_x = coords[i + 1]
            while dense[-1] != (target_y, target_x):
                curr_y, curr_x = dense[-1]
                dy = target_y - curr_y
                dx = target_x - curr_x
                options = []
                if dy != 0:
                    step_y = curr_y + (1 if dy > 0 else -1)
                    options.append((abs(dy) >= abs(dx), (step_y, curr_x)))
                if dx != 0:
                    step_x = curr_x + (1 if dx > 0 else -1)
                    options.append((abs(dx) > abs(dy), (curr_y, step_x)))
                # Prioritize move that reduces larger delta; fallback to other axis
                options.sort(reverse=True)
                moved = False
                for _, (ny, nx) in options:
                    if not (0 <= ny < max_y and 0 <= nx < max_x):
                        continue
                    if self.base_grid[ny, nx] == 1:
                        continue
                    if (ny, nx) != dense[-1]:
                        dense.append((ny, nx))
                        moved = True
                        break
                if not moved:
                    return None
        return dense

    def _sample_rrt_path(self, start, goal):
        """Sample RRT paths until a 4-connected path is produced or retries exhausted."""
        last_stats = None
        for attempt in range(self.rrt_max_retries):
            coords, stats = rrt_plan(
                self.base_grid.copy(),
                start,
                goal,
                max_iters=self.rrt_max_iters,
                step_size=self.rrt_step_size,
                goal_bias=self.rrt_goal_bias,
            )
            last_stats = stats
            if not coords:
                continue
            dense = self._densify_path(coords)
            if dense:
                return dense
        print(f"  [RRT] Agent {self.id}: failed to sample collision-free path after {self.rrt_max_retries} attempts (last stats={last_stats}).")
        return None

    def find_path(self, vertex_constraints, edge_constraints=None):
        vertex_constraints = vertex_constraints or []
        vertex_by_time = defaultdict(set)
        for y, x, t in vertex_constraints:
            vertex_by_time[int(t)].add((y, x))

        edge_constraints = edge_constraints or set()
        if not isinstance(edge_constraints, set):
            edge_constraints = set(edge_constraints)

        start = (int(self.start[0]), int(self.start[1]))
        goal = (int(self.goal[0]), int(self.goal[1]))

        dense_coords = self._sample_rrt_path(start, goal)
        if not dense_coords:
            print(f"  [RRT] Agent {self.id}: No path found by RRT after retries, waiting in place.")
            fallback = []
            for t in range(self.max_steps + 5):
                fallback.append((start[0], start[1], t))
            self.path = fallback
            return fallback
        curr_pos = dense_coords[0]
        curr_time = 0
        idx = 0
        timed_path = [(curr_pos[0], curr_pos[1], curr_time)]

        while curr_time < self.max_steps and idx < len(dense_coords) - 1:
            next_pos = dense_coords[idx + 1]
            blocked_vertex = next_pos in vertex_by_time.get(curr_time + 1, set())
            blocked_edge = (curr_pos[0], curr_pos[1], next_pos[0], next_pos[1], curr_time) in edge_constraints

            if not blocked_vertex and not blocked_edge:
                curr_time += 1
                curr_pos = next_pos
                idx += 1
                timed_path.append((curr_pos[0], curr_pos[1], curr_time))
                continue

            replanned = self._sample_rrt_path((curr_pos[0], curr_pos[1]), goal)
            if replanned and len(replanned) > 1:
                dense_coords = replanned
                idx = 0
                continue

            wait_blocked = curr_pos in vertex_by_time.get(curr_time + 1, set())
            if wait_blocked:
                print(f"  [RRT] Agent {self.id} t={curr_time}: forced WAIT at ({curr_pos[0]},{curr_pos[1]}) despite vertex constraint.")
            curr_time += 1
            timed_path.append((curr_pos[0], curr_pos[1], curr_time))

        last_y, last_x, last_t = timed_path[-1]
        pad_limit = self.max_steps + 5
        for t in range(last_t + 1, pad_limit):
            timed_path.append((last_y, last_x, t))

        self.path = timed_path
        return timed_path


def generate_random_grid(rng, map_size, obstacle_density, min_free_cells):
    """Generate a random occupancy grid that leaves enough free cells."""
    H, W = map_size
    total_cells = H * W
    target_free = max(min_free_cells, total_cells // 5)
    while True:
        grid = (rng.random((H, W)) < obstacle_density).astype(int)
        free_cells = np.sum(grid == 0)
        if free_cells >= target_free:
            return grid


def sample_agent_configs(rng, grid, num_agents):
    """Sample unique start/goal pairs for each agent on free cells."""
    free_cells = np.argwhere(grid == 0)
    needed = num_agents * 2
    if len(free_cells) < needed:
        raise ValueError("Not enough free cells to place agent start/goal pairs.")
    perm = rng.permutation(len(free_cells))[:needed]
    coords = [tuple(map(int, free_cells[idx])) for idx in perm]
    color_cycle = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]
    configs = []
    for agent_id in range(num_agents):
        start = coords[agent_id]
        goal = coords[agent_id + num_agents]
        configs.append({
            "id": agent_id,
            "start": start,
            "goal": goal,
            "color": color_cycle[agent_id % len(color_cycle)]
        })
    return configs


def format_save_path(base_path, run_idx):
    if not base_path:
        return None
    root, ext = os.path.splitext(base_path)
    return f"{root}_run{run_idx}{ext}"

class PBSNode:
    """
    Node in the PBS High-Level Search Tree.
    Contains partial ordering constraints and the current solution plan.
    High level code taken from https://github.com/Jiaoyang-Li/PBS/tree/master/src
    """
    def __init__(self, priorities=None, solution=None, cost=0, parent=None, node_id=0):
        # priorities: Set of tuples (high_id, low_id) -> high_id has priority over low_id
        # unset when you make root node
        self.priorities = set(priorities) if priorities else set()
        # solution: dict {agent_id: path}
        self.solution = solution if solution else {}
        self.cost = cost
        self.conflicts = []
        self.parent = parent
        self.node_id = node_id
        self.children = []
        self.collision_info = None  # (a1, a2, loc, t) that caused this branch
        self.added_priority = None  
    
    def __lt__(self, other):
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
        
        # Tree tracking
        self.root_node = None
        self.all_nodes = []
        self.solution_node = None

    def add_agent(self, agent_id, model_path, start_pos, goal_pos, color):
        model = DQN.load(model_path)
        agent = Agent(agent_id, model, start_pos, goal_pos, color, 
                      self.map_size, self.obstacles, self.max_steps)
        self.agents[agent_id] = agent
        self.agent_ids.append(agent_id)

    def add_rrt_agent(
        self,
        agent_id,
        start_pos,
        goal_pos,
        color,
        rrt_max_iters=4000,
        rrt_step_size=1,
        rrt_goal_bias=0.1,
        rrt_max_retries=10,
    ):
        agent = RRTAgent(
            agent_id,
            start_pos,
            goal_pos,
            color,
            self.map_size,
            self.obstacles,
            self.max_steps,
            rrt_max_iters=rrt_max_iters,
            rrt_step_size=rrt_step_size,
            rrt_goal_bias=rrt_goal_bias,
            rrt_max_retries=rrt_max_retries,
        )
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
        """
        order = self.topological_sort(node.priorities)
        if order is None:
            return False  # Invalid priority (cycle)

        vertex_constraints = set()
        edge_constraints = set()
        
        for agent_id in order:
            agent = self.agents[agent_id]
            
            # KEY CHANGE: At root (no priorities), plan WITHOUT constraints
            # This allows collisions to be detected, triggering PBS branching
            if len(node.priorities) == 0:
                # Root node: plan independently (no constraints from other agents)
                if agent_id in agents_to_replan or agent_id not in node.solution:
                    path = agent.find_path([], set())  # Empty constraints!
                    node.solution[agent_id] = path
            else:
                # Non-root: respect priority ordering
                if agent_id in agents_to_replan or agent_id not in node.solution:
                    path = agent.find_path(list(vertex_constraints), edge_constraints)
                    node.solution[agent_id] = path
            
            # Add this agent's path to constraints for subsequent agents
            current_path = node.solution[agent_id]
            for i, (y, x, t) in enumerate(current_path):
                vertex_constraints.add((y, x, t))
                if i + 1 < len(current_path):
                    ny, nx, nt = current_path[i + 1]
                    if (ny, nx) != (y, x):
                        edge_constraints.add((ny, nx, y, x, t))
                
        node.cost = sum(len(path) for path in node.solution.values())
        self._print_paths_by_timestep(node.solution)
        
        return True

    def _print_paths_by_timestep(self, solution, max_timesteps=20):
        """Print all agents' positions grouped by timestep."""
        if not solution:
            return
        
        max_len = min(max(len(path) for path in solution.values()), max_timesteps)
        action_names = {(0,0): "WAIT", (-1,0): "UP", (1,0): "DOWN", (0,-1): "LEFT", (0,1): "RIGHT"}
        
        print("\n--- Paths by Timestep ---")
        for t in range(max_len):
            print(f"t={t}:")
            for agent_id in sorted(solution.keys()):
                path = solution[agent_id]
                if t < len(path):
                    y, x, _ = path[t]
                    # Determine move direction
                    if t + 1 < len(path):
                        ny, nx, _ = path[t + 1]
                        dy, dx = ny - y, nx - x
                        move = action_names.get((dy, dx), "?")
                        print(f"    Agent {agent_id}: ({y},{x}) -> ({ny},{nx}) [{move}]")
                    else:
                        print(f"    Agent {agent_id}: ({y},{x}) [DONE]")
        print("-------------------------\n")

    def find_collisions(self, solution):
      """
      Find the first collision between any two agents.
      Returns: (agent_i, agent_j, location, time) or None
      """
      if not solution:
          return None
      
      # Find maximum path length to check all timesteps
      max_len = max(len(path) for path in solution.values())
      
      # Check each timestep for vertex collisions
      for t in range(max_len):
          pos_at_time = {}  # Maps (y, x) -> agent_id at this timestep
          
          for agent_id, path in solution.items():
              # Get agent position at time t
              if t < len(path):
                  y, x, time_coord = path[t]
                  pos = (y, x)
              else:
                  # Agent finished - shouldn't happen with proper padding, 
                  # but handle gracefully
                  continue
              
              # Check if another agent is at same position at same time
              if pos in pos_at_time:
                  other_agent = pos_at_time[pos]
                  print(f"  -> Vertex collision detected: Agent {agent_id} and {other_agent} at {pos} at t={t}")
                  return (agent_id, other_agent, pos, t)
              
              pos_at_time[pos] = agent_id
      
      # Check edge collisions (agents swapping positions)
      for t in range(max_len - 1):
          for agent_i, path_i in solution.items():
              for agent_j, path_j in solution.items():
                  if agent_i >= agent_j:  # Only check each pair once
                      continue
                  
                  if t >= len(path_i) - 1 or t >= len(path_j) - 1:
                      continue
                  
                  # Position at time t
                  pos_i_t = (path_i[t][0], path_i[t][1])
                  pos_j_t = (path_j[t][0], path_j[t][1])
                  
                  # Position at time t+1
                  pos_i_t1 = (path_i[t+1][0], path_i[t+1][1])
                  pos_j_t1 = (path_j[t+1][0], path_j[t+1][1])
                  
                  # Check if agents swapped positions (edge collision)
                  if pos_i_t == pos_j_t1 and pos_j_t == pos_i_t1:
                      print(f"  -> Edge collision detected: Agent {agent_i} and {agent_j} swapping {pos_i_t}<->{pos_j_t1} at t={t}")
                      return (agent_i, agent_j, pos_i_t, t)
      
      return None

    def solve(self):
        print(f"Starting PBS for {len(self.agents)} agents...")
        
        # Reset tree tracking
        self.all_nodes = []
        self.solution_node = None
        
        # 1. Root Node
        root = PBSNode(node_id=0)
        self.root_node = root
        self.all_nodes.append(root)
        
        # Initially, plan all agents (empty priorities -> order doesn't matter yet, 
        # but topological sort will give default order)
        success = self.update_plan(root, set(self.agent_ids))
        if not success: return None
        
        # 2. Check root conflicts
        collision = self.find_collisions(root.solution)
        if collision is None:
            self.solution_node = root
            self.print_pbs_tree()
            return root.solution
            
        root.conflicts = [collision]
        root.collision_info = collision
        
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
                self.solution_node = curr_node
                self.print_pbs_tree()
                return curr_node.solution
            
            a1, a2, loc, t = collision
            curr_node.collision_info = collision
            print(f"Node {curr_node.node_id}: Collision {a1}-{a2} at {loc} t={t}")
            
            # Branching: Resolve (a1, a2) collision
            # Option 1: a1 > a2 (a1 has priority)
            child1 = PBSNode(
                priorities=copy.deepcopy(curr_node.priorities), 
                solution=copy.deepcopy(curr_node.solution),
                parent=curr_node,
                node_id=len(self.all_nodes)
            )
            child1.priorities.add((a1, a2))
            child1.added_priority = (a1, a2)
            
            # Only replan a2 (the lower priority one)
            if self.update_plan(child1, {a2}):
                curr_node.children.append(child1)
                self.all_nodes.append(child1)
                open_list.append(child1)
                self.generated_nodes += 1
                
            # Option 2: a2 > a1 (a2 has priority)
            child2 = PBSNode(
                priorities=copy.deepcopy(curr_node.priorities), 
                solution=copy.deepcopy(curr_node.solution),
                parent=curr_node,
                node_id=len(self.all_nodes)
            )
            child2.priorities.add((a2, a1))
            child2.added_priority = (a2, a1)
            
            # Only replan a1
            if self.update_plan(child2, {a1}):
                curr_node.children.append(child2)
                self.all_nodes.append(child2)
                open_list.append(child2)
                self.generated_nodes += 1
                
        print("PBS failed to find a solution.")
        self.print_pbs_tree()
        return None

    def print_pbs_tree(self):
        """Print the PBS search tree structure."""
        print("\n" + "=" * 60)
        print("PBS SEARCH TREE")
        print("=" * 60)
        
        if not self.root_node:
            print("  (empty tree)")
            return
        
        def print_node(node, prefix="", is_last=True):
            connector = "└── " if is_last else "├── "
            
            # Node info
            node_str = f"Node {node.node_id}"
            if node.added_priority:
                high, low = node.added_priority
                node_str += f" [Added: {high}>{low}]"
            
            # Collision info
            if node.collision_info:
                a1, a2, loc, t = node.collision_info
                node_str += f" | Collision: {a1}-{a2} at {loc} t={t}"
            
            # Solution status
            if node == self.solution_node:
                node_str += " ✓ SOLUTION"
            
            # Cost
            node_str += f" | Cost: {node.cost}"
            
            # Priorities
            node_str += f" | Priorities: {{{', '.join(f'{h}>{l}' for h, l in sorted(node.priorities))}}}"
            
            print(prefix + connector + node_str)
            
            # Print children
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(node.children):
                print_node(child, child_prefix, i == len(node.children) - 1)
        
        print_node(self.root_node)
        
        # Summary
        print("\n" + "-" * 60)
        print(f"Total nodes generated: {self.generated_nodes}")
        print(f"Total nodes expanded:  {self.expanded_nodes}")
        if self.solution_node:
            print(f"Solution found at Node {self.solution_node.node_id}")
            print(f"Final priorities: {{{', '.join(f'{h}>{l}' for h, l in sorted(self.solution_node.priorities))}}}")
        else:
            print("No solution found")
        print("=" * 60 + "\n")

    def print_priority_chain(self):
        """Print the path from root to solution showing how priorities were added."""
        if not self.solution_node:
            print("No solution to trace.")
            return
        
        print("\n" + "=" * 60)
        print("PRIORITY CHAIN (Root -> Solution)")
        print("=" * 60)
        
        # Trace path from solution back to root
        path = []
        node = self.solution_node
        while node:
            path.append(node)
            node = node.parent
        path.reverse()
        
        for i, node in enumerate(path):
            indent = "  " * i
            if node.parent is None:
                print(f"{indent}Root (Node 0)")
                if node.collision_info:
                    a1, a2, loc, t = node.collision_info
                    print(f"{indent}  └─ Collision: Agent {a1} vs {a2} at {loc}, t={t}")
            else:
                high, low = node.added_priority
                print(f"{indent}└─ Node {node.node_id}: Added {high} > {low}")
                if node.collision_info and node != self.solution_node:
                    a1, a2, loc, t = node.collision_info
                    print(f"{indent}    └─ Collision: Agent {a1} vs {a2} at {loc}, t={t}")
                elif node == self.solution_node:
                    print(f"{indent}    └─ No collisions - SOLUTION FOUND")
        
        print("=" * 60 + "\n")

    def _completion_time(self, agent_id, path):
        """Return first timestep where agent reaches its goal (fallback to end)."""
        goal = tuple(self.agents[agent_id].goal)
        for idx, (y, x, _) in enumerate(path):
            if (y, x) == goal:
                return idx
        return len(path) - 1

    def visualize_solution(self, solution, animate=True, delay=0.1, save_path=None, save_gif=None):
        """Visualizes the found solution.
        
        Args:
            solution: Dict of {agent_id: path}
            animate: Whether to animate the solution
            delay: Delay between frames in seconds
            save_path: Path to save final frame as image
            save_gif: Path to save animation as GIF (e.g., 'solution.gif')
        """
        if not solution:
            print("No solution to visualize.")
            return

        # Switch to Agg backend when saving GIFs to avoid display issues
        original_backend = None
        if save_gif:
            original_backend = matplotlib.get_backend()
            plt.switch_backend('Agg')
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Calculate max time until all agents reach their goals
        max_time = max(self._completion_time(agent_id, path) for agent_id, path in solution.items())
        
        frames = []  # Store frames for GIF
        
        if animate and not save_gif:
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
            
            # Capture frame for GIF using savefig to buffer
            if save_gif:
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                try:
                    from PIL import Image
                    image = Image.open(buf)
                    frames.append(image.copy())
                    image.close()
                except ImportError:
                    print("Error: 'Pillow' package required for GIF export.")
                    print("Install with: pip install Pillow")
                    return
                buf.close()
            
            if animate and not save_gif:
                plt.draw()
                plt.pause(delay)
            elif timestep == 0 and not save_gif:
                plt.show()
        
        # Save GIF
        if save_gif and frames:
            # Convert delay from seconds to milliseconds
            duration_ms = int(delay * 1000)
            frames[0].save(
                save_gif,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0
            )
            print(f"GIF saved to: {save_gif}")
                
        if save_path:
            plt.savefig(save_path)
            print(f"Image saved to: {save_path}")
            
        if animate and not save_gif:
            plt.ioff()
            plt.show()
        
        plt.close(fig)
        
        if original_backend:
            plt.switch_backend(original_backend)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='Path to DQN model (required unless --use_rrt)')
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--no_anim', action='store_true')
    parser.add_argument('--save_gif', type=str, default=None, help='Path to save animation as GIF')
    parser.add_argument('--save_img', type=str, default=None, help='Path to save final frame as image')
    parser.add_argument('--use_rrt', action='store_true', help='Use RRT low-level agent instead of DQN policy')
    parser.add_argument('--rrt_max_iters', type=int, default=4000, help='RRT max iterations (when --use_rrt)')
    parser.add_argument('--rrt_step_size', type=int, default=1, help='RRT step size (when --use_rrt)')
    parser.add_argument('--rrt_goal_bias', type=float, default=0.1, help='RRT goal bias probability (when --use_rrt)')
    parser.add_argument('--rrt_max_retries', type=int, default=40, help='RRT resampling attempts for collision-free discrete path')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of randomized scenarios to solve')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--obstacle_density', type=float, default=0.15, help='Density of obstacles in randomized grids (0-1)')
    args = parser.parse_args()

    if not args.use_rrt and not args.model:
        parser.error("--model is required unless --use_rrt is specified")
    if args.num_runs <= 0:
        parser.error("--num_runs must be at least 1")

    map_size = (20, 20)
    num_agents = 4
    rng = np.random.default_rng(args.seed)
    success_runs = 0
    
    print("=" * 50)
    print("PBS RANDOM SCENARIOS")
    print("=" * 50)
    print(f"Running {args.num_runs} randomized 20x20 grids "
          f"(obstacle_density={args.obstacle_density:.2f})\n")

    for run_idx in range(1, args.num_runs + 1):
        print(f"\n--- Random Run {run_idx}/{args.num_runs} ---")
        grid = generate_random_grid(rng, map_size, args.obstacle_density, min_free_cells=num_agents * 4)
        obstacle_coords = [tuple(map(int, coord)) for coord in np.argwhere(grid == 1)]
        obstacle_count = len(obstacle_coords)
        agent_configs = sample_agent_configs(rng, grid, num_agents)

        planner = PBSPlanner(map_size=map_size, obstacles=obstacle_coords, max_steps=args.max_steps)

        print(f"Obstacle cells: {obstacle_count} / {map_size[0] * map_size[1]}")
        print("Agent Setup:")
        for cfg in agent_configs:
            if args.use_rrt:
                planner.add_rrt_agent(
                    cfg["id"],
                    cfg["start"],
                    cfg["goal"],
                    cfg["color"],
                    rrt_max_iters=args.rrt_max_iters,
                    rrt_step_size=args.rrt_step_size,
                    rrt_goal_bias=args.rrt_goal_bias,
                    rrt_max_retries=args.rrt_max_retries,
                )
            else:
                planner.add_agent(cfg["id"], args.model, cfg["start"], cfg["goal"], cfg["color"])
            print(f"  Agent {cfg['id']}: {cfg['start']} -> {cfg['goal']} ({cfg['color']})")

        start_time = time.time()
        solution = planner.solve()
        duration = time.time() - start_time

        print("\n" + "-" * 50)
        if solution:
            success_runs += 1
            print(f"Run {run_idx}: SOLVED in {duration:.2f}s")
            print(f"Nodes Generated: {planner.generated_nodes}")
            print(f"Nodes Expanded:  {planner.expanded_nodes}")

            print("\nFinal path lengths:")
            for agent_id, path in solution.items():
                completion = planner._completion_time(agent_id, path)
                print(f"  Agent {agent_id}: {completion} steps")

            gif_path = format_save_path(args.save_gif, run_idx)
            img_path = format_save_path(args.save_img, run_idx)
            planner.visualize_solution(
                solution,
                animate=not args.no_anim,
                delay=0.3,
                save_gif=gif_path,
                save_path=img_path
            )
        else:
            print(f"Run {run_idx}: NO SOLUTION FOUND")
        print("-" * 50)

    print("\n" + "=" * 50)
    print(f"Completed {args.num_runs} runs | Successes: {success_runs} | Failures: {args.num_runs - success_runs}")
    print("=" * 50)

if __name__ == "__main__":
    main()
