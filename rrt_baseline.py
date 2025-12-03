"""
RRT Baseline Planner
Single-agent path planning baseline for comparison with PBS and PBS+RL.

How to run: python rrt_baseline.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

Coord = Tuple[int, int]


class Node:
    def __init__(self, pos: Coord, parent: Optional[int]):
        self.pos = pos
        self.parent = parent


"""
Plan a path from start to goal using RRT. 
Returns the path and statistics.
"""
def rrt_plan(
    grid: np.ndarray,
    start: Coord,
    goal: Coord,
    max_iters: int = 5000,
    step_size: int = 1,
    goal_bias: float = 0.1,
) -> Tuple[List[Coord], Dict]:

    rows, cols = grid.shape
    
    # validate input
    if grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
        return [], {"success": False, "iterations": 0, "time": 0.0, "nodes": 0, "path_length": 0}
    if start == goal:
        return [start], {"success": True, "iterations": 0, "time": 0.0, "nodes": 1, "path_length": 1}
    
    # initialize RRT tree and start time
    nodes: List[Node] = [Node(start, parent=None)]
    t_start = time.perf_counter()
    
    # sample -> extend -> check -> add
    for iteration in range(max_iters):
        
        # sample (with goal bias)
        if np.random.random() < goal_bias:
            q_rand = goal
        else:
            while True:
                r, c = np.random.randint(0, rows), np.random.randint(0, cols)

                # reject if sample is an obstacle
                if grid[r, c] == 0:
                    q_rand = (r, c)
                    break
        
        # find nearest node
        best_idx, best_dist = 0, float("inf")
        for i, node in enumerate(nodes):

            # euclidean distance
            d = (node.pos[0] - q_rand[0])**2 + (node.pos[1] - q_rand[1])**2
            if d < best_dist:
                best_dist, best_idx = d, i
        q_near = nodes[best_idx].pos
        
        # extend toward sample
        dr, dc = q_rand[0] - q_near[0], q_rand[1] - q_near[1]
        dist = max(1, int(np.hypot(dr, dc))) # avoid 0
        step = min(step_size, dist)
        q_new = (int(round(q_near[0] + dr/dist * step)),
                 int(round(q_near[1] + dc/dist * step)))
        
        # check bounds
        if not (0 <= q_new[0] < rows and 0 <= q_new[1] < cols):
            continue
        
        # check obstacle 
        if grid[q_new[0], q_new[1]] == 1:
            continue
        
        # collision check along edge (interpolation)
        num_steps = max(abs(q_new[0] - q_near[0]), abs(q_new[1] - q_near[1]))
        collision = False
        for i in range(num_steps + 1):
            t = i / max(1, num_steps)
            r = int(round(q_near[0] + t * (q_new[0] - q_near[0])))
            c = int(round(q_near[1] + t * (q_new[1] - q_near[1])))
            if grid[r, c] == 1:
                collision = True # edge is obstructed
                break
        if collision:
            continue
        
        # add node to tree
        nodes.append(Node(q_new, parent=best_idx))
        
        # check if goal reached
        if q_new == goal:
            path, idx = [], len(nodes) - 1 # reconstruct path from goal to start
            while idx is not None:
                path.append(nodes[idx].pos)
                idx = nodes[idx].parent
            path.reverse() # reverse path to start to goal
            elapsed = time.perf_counter() - t_start
            return path, {
                "success": True,
                "iterations": iteration + 1,
                "time": elapsed,
                "nodes": len(nodes),
                "path_length": len(path),
            }
    # no path found (reach max iterations)
    elapsed = time.perf_counter() - t_start
    return [], {
        "success": False,
        "iterations": max_iters,
        "time": elapsed,
        "nodes": len(nodes),
        "path_length": 0,
    }

"""
Visualize RRT solution (similar stype to PBS)
"""
def visualize_rrt(
    grid: np.ndarray,
    start: Coord,
    goal: Coord,
    path: List[Coord],
    title: str = "RRT Path",
) -> None:

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # draw grid using imshow (same as PBS)
    ax.imshow(grid, cmap='gray_r', origin='upper', vmin=0, vmax=1)
    
    # draw path (blue line / similar to PBS)
    if path:
        rows_path = [r for (r, c) in path]
        cols_path = [c for (r, c) in path]
        ax.plot(cols_path, rows_path, '-o', color='blue', linewidth=2, 
                markersize=4, markerfacecolor='white', markeredgecolor='blue',
                label='RRT Path', zorder=3)
    
    # draw start (green circle)
    ax.scatter(start[1], start[0], color='green', s=200, edgecolors='black', 
               linewidths=2, zorder=5, label='Start')
    
    # draw goal (gold star)
    ax.scatter(goal[1], goal[0], marker='*', color='gold', s=400,
               edgecolors='black', linewidths=1.5, zorder=4, label='Goal')
    
    # grid lines (same as PBS)
    ax.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


"""
Demo with a 20x20 grid with obstacles in the center
"""
if __name__ == "__main__":
    print("=" * 50)
    print("RRT Baseline - Single Agent Path Planning")
    print("=" * 50)
    
    # create test grid
    grid = np.zeros((20, 20), dtype=int)
    
    # add obstacles
    # border walls
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    center = 10
    grid[center-4:center+5, center] = 1
    grid[center, center-4:center+5] = 1
    
    start = (2, 2)
    goal = (17, 17)
    
    print(f"Grid: 20x20")
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"Obstacles: {np.sum(grid == 1)} cells")
    print("-" * 50)
    
    # run rrt
    path, stats = rrt_plan(grid, start, goal, max_iters=5000, step_size=2)
    
    # print results 
    print(f"Success: {stats['success']}")
    print(f"Planning time: {stats['time']:.4f} seconds")
    print(f"Path length: {stats['path_length']} waypoints")
    print(f"Tree nodes explored: {stats['nodes']}")
    print(f"Iterations: {stats['iterations']}")
    print("=" * 50)
    
    # visualize
    if path:
        visualize_rrt(grid, start, goal, path,
            title=f"RRT Baseline | Path: {len(path)} waypoints | Time: {stats['time']:.3f}s")
    else:
        print("No path found!")
