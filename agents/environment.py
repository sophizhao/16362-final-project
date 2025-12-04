# path_planning_mask_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque


# Much of this code is adapted from my summer research with SBPL 
class PathPlanningMaskEnv(gym.Env):
    """
    Simplified gridworld for DQN path planning:
      - Grid with random obstacles
      - Simple 5-action movement (stay, up, down, left, right)
      - LiDAR scans
      - Distance-based potential field
      - Action history
      - Distance gradient field
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, map_size=(10, 10), obstacles=None, max_steps=200, hist_len=6):
        super().__init__()

        self.H, self.W = map_size
        self.max_steps = max_steps
        self.hist_len = hist_len
        self.step_count = 0

        # Map / obstacles
        self.map = np.zeros((self.H, self.W), dtype=np.uint8)
        if obstacles is not None:
            for (y, x) in obstacles:
                if 0 <= y < self.H and 0 <= x < self.W:
                    self.map[y, x] = 1

        # Action space: 0=stay, 1=up, 2=down, 3=left, 4=right
        self.num_actions = 5
        self.action_space = spaces.Discrete(self.num_actions)

        self.lidar_beams = 16
        self.max_scan_dist = 20.0

        self.observation_space = spaces.Dict(
            {
                "scan": spaces.Box(0.0, self.max_scan_dist, (self.lidar_beams,), dtype=np.float32),
                "goal_vec": spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32),
                "dist_grad": spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
                "dist_phi": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
                "hist": spaces.MultiDiscrete([self.num_actions] * hist_len),
            }
        )

        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.goal = np.array([self.H - 1, self.W - 1], dtype=np.int32)
        self.action_history = deque(maxlen=hist_len)

        self.time_cost = -0.1
        self.completion_bonus = 10.0
        self.collision_penalty = -10.0

        # Distance field and gradient (computed on reset)
        self.dist_map = None
        self.grad_x = None
        self.grad_y = None
        self.max_potential = 0.0


    def _compute_distance_map(self, goal_pos):
        """
        BFS to compute distance from every free cell to goal.
        Returns distance map in grid coordinates.
        """
        from collections import deque as bfs_deque

        dist = np.full((self.H, self.W), np.inf, dtype=np.float32)
        free = (self.map == 0)

        gy, gx = goal_pos
        if not free[gy, gx]:
            # If goal is on obstacle, find nearest free cell
            visited = np.zeros_like(free, dtype=bool)
            q = bfs_deque([(gy, gx)])
            visited[gy, gx] = True
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            while q:
                y0, x0 = q.popleft()
                for dy, dx in neighbors:
                    yn, xn = y0 + dy, x0 + dx
                    if 0 <= yn < self.H and 0 <= xn < self.W and not visited[yn, xn]:
                        if free[yn, xn]:
                            gy, gx = yn, xn
                            q.clear()
                            break
                        visited[yn, xn] = True
                        q.append((yn, xn))
        dist[gy, gx] = 0.0
        q = bfs_deque([(gy, gx)])
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while q:
            y0, x0 = q.popleft()
            for dy, dx in neighbors:
                yn, xn = y0 + dy, x0 + dx
                if (
                    0 <= yn < self.H
                    and 0 <= xn < self.W
                    and free[yn, xn]
                    and not np.isfinite(dist[yn, xn])
                ):
                    dist[yn, xn] = dist[y0, x0] + 1.0
                    q.append((yn, xn))

        return dist

    def _compute_gradient_field(self):
        """
        Compute normalized gradient of distance field.
        Points toward decreasing distance (toward goal).
        """
        gy, gx = np.gradient(self.dist_map)
        mag = np.hypot(gx, gy)

        self.grad_x = np.zeros_like(gx, dtype=np.float32)
        self.grad_y = np.zeros_like(gy, dtype=np.float32)

        nz = mag > 1e-8
        self.grad_x[nz] = -gx[nz] / mag[nz]
        self.grad_y[nz] = -gy[nz] / mag[nz]

        # Zero out gradients on obstacles and unreachable cells
        valid = (self.map == 0) & np.isfinite(self.dist_map)
        self.grad_x[~valid] = 0.0
        self.grad_y[~valid] = 0.0

    def _compute_lidar(self):
        """
        Cast rays uniformly in [0, 2π) from agent position.
        Returns normalized distances to nearest obstacle or boundary.
        """
        angles = np.linspace(0, 2 * np.pi, self.lidar_beams, endpoint=False)
        readings = []

        for theta in angles:
            dist = self.max_scan_dist
            for r in range(1, int(self.max_scan_dist) + 1):
                y = int(self.agent_pos[0] + r * np.sin(theta))
                x = int(self.agent_pos[1] + r * np.cos(theta))

                if not (0 <= y < self.H and 0 <= x < self.W):
                    dist = r
                    break
                if self.map[y, x] == 1:
                    dist = r
                    break

            readings.append(dist / self.max_scan_dist)

        return np.array(readings, dtype=np.float32)

    def _get_distance_value(self, pos):
        """Get interpolated distance value at position."""
        y, x = pos
        y = np.clip(y, 0, self.H - 1)
        x = np.clip(x, 0, self.W - 1)
        return float(self.dist_map[int(y), int(x)])

    def _get_gradient_at_pos(self, pos):
        """Get gradient vector at position."""
        y, x = pos
        y = int(np.clip(y, 0, self.H - 1))
        x = int(np.clip(x, 0, self.W - 1))
        return np.array([self.grad_x[y, x], self.grad_y[y, x]], dtype=np.float32)

    def _get_free_cells(self):
        """Get all free (non-obstacle) cells."""
        return np.argwhere(self.map == 0)

    # ============================================================
    # ---------------------- Gym Methods -------------------------
    # ============================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.action_history.clear()
        for _ in range(self.hist_len):
            self.action_history.append(0)

        # Get all free cells
        free_cells = self._get_free_cells()

        if len(free_cells) < 2:
            # Fallback if not enough free cells
            self.agent_pos = np.array([0, 0], dtype=np.int32)
            self.goal = np.array([self.H - 1, self.W - 1], dtype=np.int32)
        else:
            # Randomly select start and goal from free cells
            indices = self.np_random.choice(len(free_cells), size=2, replace=False)
            self.agent_pos = free_cells[indices[0]].copy()
            self.goal = free_cells[indices[1]].copy()

        # Compute distance field and gradient from goal
        self.dist_map = self._compute_distance_map(self.goal)
        reachable = np.isfinite(self.dist_map)
        self.max_potential = float(np.max(self.dist_map[reachable])) if np.any(reachable) else 1.0
        self.dist_map[~reachable] = self.max_potential

        self._compute_gradient_field()

        obs = self._compute_obs()
        return obs, {}

    def _compute_obs(self):
        """Compute observation dictionary."""
        # LiDAR scan
        scan = self._compute_lidar()

        # Goal vector (relative position to goal)
        goal_vec = (self.goal - self.agent_pos).astype(np.float32)

        # Distance gradient at agent position
        dist_grad = self._get_gradient_at_pos(self.agent_pos)

        # Normalized distance to goal
        dist_val = self._get_distance_value(self.agent_pos)
        dist_phi = np.array([dist_val / (self.max_potential + 1e-6)], dtype=np.float32)

        # Action history
        hist = np.array(self.action_history, dtype=np.int64)

        return {
            "scan": scan,
            "goal_vec": goal_vec,
            "dist_grad": dist_grad,
            "dist_phi": dist_phi,
            "hist": hist,
        }

    def step(self, action):
        self.step_count += 1

        # Store old distance for reward shaping
        old_dist = self._get_distance_value(self.agent_pos)

        # Define movements
        moves = {
            0: (0, 0),    # stay
            1: (-1, 0),   # up
            2: (1, 0),    # down
            3: (0, -1),   # left
            4: (0, 1),    # right
        }

        dy, dx = moves[action]
        new_y = self.agent_pos[0] + dy
        new_x = self.agent_pos[1] + dx

        # Check if move is valid
        valid = (
            0 <= new_y < self.H
            and 0 <= new_x < self.W
            and self.map[new_y, new_x] == 0
        )

        if not valid:
            # Invalid action - collision or out of bounds
            reward = self.collision_penalty + self.time_cost
            self.action_history.append(action)
        else:
            # Valid action - move agent
            self.agent_pos[:] = (new_y, new_x)
            self.action_history.append(action)

            # Calculate new distance
            new_dist = self._get_distance_value(self.agent_pos)

            # Reward with distance-based shaping
            reward = self.time_cost + (old_dist - new_dist)

        # Check if goal reached
        reached = np.array_equal(self.agent_pos, self.goal)
        if reached:
            reward += self.completion_bonus

        # Check termination conditions
        terminated = reached
        truncated = self.step_count >= self.max_steps

        info = {"reached": reached, "collision": not valid}

        return self._compute_obs(), reward, terminated, truncated, info

    def render(self, mode='human'):
        """Matplotlib-based rendering."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle
        
        if not hasattr(self, '_fig') or self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(8, 8))
            plt.ion()  # Interactive mode
        
        ax = self._ax
        ax.clear()
        
        # Draw grid
        for y in range(self.H + 1):
            ax.plot([0, self.W], [y, y], 'k-', linewidth=0.5, alpha=0.3)
        for x in range(self.W + 1):
            ax.plot([x, x], [0, self.H], 'k-', linewidth=0.5, alpha=0.3)
        
        # Draw obstacles
        for y in range(self.H):
            for x in range(self.W):
                if self.map[y, x] == 1:
                    rect = Rectangle((x, y), 1, 1, facecolor='black', edgecolor='gray')
                    ax.add_patch(rect)
        
        # Draw distance field as heatmap
        if self.dist_map is not None:
            normalized_dist = self.dist_map / (self.max_potential + 1e-6)
            ax.imshow(
                normalized_dist,
                origin='lower',
                extent=[0, self.W, 0, self.H],
                cmap='viridis',
                alpha=0.3,
                interpolation='bilinear'
            )
        
        if self.grad_x is not None and self.grad_y is not None:
            step = 2
            for y in range(0, self.H, step):
                for x in range(0, self.W, step):
                    if self.map[y, x] == 0 and np.isfinite(self.dist_map[y, x]):
                        gx = self.grad_x[y, x]
                        gy = self.grad_y[y, x]
                        mag = np.hypot(gx, gy)
                        if mag > 0.1:
                            ax.arrow(
                                x + 0.5, y + 0.5,
                                gx * 0.4, gy * 0.4,
                                head_width=0.15,
                                head_length=0.1,
                                fc='orange',
                                ec='orange',
                                alpha=0.6,
                                linewidth=1
                            )
        
        # Draw goal
        goal_circle = Circle(
            (self.goal[1] + 0.5, self.goal[0] + 0.5),
            0.4,
            facecolor='green',
            edgecolor='darkgreen',
            linewidth=2,
            alpha=0.8,
            zorder=5
        )
        ax.add_patch(goal_circle)
        ax.text(
            self.goal[1] + 0.5, self.goal[0] + 0.5,
            'G',
            ha='center', va='center',
            color='white',
            fontsize=14,
            fontweight='bold',
            zorder=6
        )
        
        # Draw agent
        agent_circle = Circle(
            (self.agent_pos[1] + 0.5, self.agent_pos[0] + 0.5),
            0.35,
            facecolor='blue',
            edgecolor='darkblue',
            linewidth=2,
            alpha=0.9,
            zorder=5
        )
        ax.add_patch(agent_circle)
        ax.text(
            self.agent_pos[1] + 0.5, self.agent_pos[0] + 0.5,
            'A',
            ha='center', va='center',
            color='white',
            fontsize=12,
            fontweight='bold',
            zorder=6
        )
        
        # Set limits and labels
        ax.set_xlim(0, self.W)
        ax.set_ylim(0, self.H)
        ax.set_aspect('equal')
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        
        # Title with info
        dist_to_goal = np.linalg.norm(self.agent_pos - self.goal)
        ax.set_title(
            f'Step: {self.step_count}/{self.max_steps} | '
            f'Agent: {tuple(self.agent_pos)} | Goal: {tuple(self.goal)} | '
            f'Distance: {dist_to_goal:.2f}',
            fontsize=11,
            pad=10
        )
        
        plt.tight_layout()
        
        if mode == 'human':
            plt.draw()
            plt.pause(0.001)
        
        return self._fig