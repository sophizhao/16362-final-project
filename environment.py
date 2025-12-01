# planning_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Optional, Dict, Tuple
import matplotlib
from matplotlib.patches import Circle
matplotlib.use("Agg")

from domains_cc.map_and_scen_utils import parse_map_file, parse_scen_file
from domains_cc.benchmark.parallel_benchmark_base import get_map_file_path
from domains_cc.footprints.footprint import load_footprint_from_yaml_path
from domains_cc.dynamics.robot_dynamics import load_dynamics_from_yaml_path
from domains_cc.worldCC import WorldCollisionChecker
from domains_cc.worldCCVisualizer import addGridToPlot, addXYThetaToPlot
from domains_cc.worldCC_CBS import WorldConstraint


class PathPlanningMaskEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        scen_file: str,
        problem_index: int,
        dynamics_config: str,
        footprint_config: str,
        max_steps: int = 1000,
        angle_tol: float = 0.10,
        n_lasers: int = 16,
        max_scan_dist: float = 10.0,
        hist_len: int = 6,
    ):
        super().__init__()

        # --- dynamics & footprint ---
        self.dynamics, _ = load_dynamics_from_yaml_path(dynamics_config)
        self.footprint, _, _ = load_footprint_from_yaml_path(footprint_config)

        # --- load raw map (top-origin) ---
        map_path = get_map_file_path(scen_file)
        raw_grid, resolution = parse_map_file(map_path)
        self.resolution = resolution
        self.max_scan_dist = max_scan_dist

        self.world_cc = WorldCollisionChecker(raw_grid, resolution)
        # raw grid is accessed x, y
        # i want to access bottom_grid as row, col
        bottom_grid = np.transpose(raw_grid)
        # print("bottom_grid shape: ", bottom_grid.shape)
        free_mask   = (~bottom_grid).astype(np.uint8)


        # --- start / goal (world coords, origin bottom-left) ---
        pairs = parse_scen_file(scen_file)
        self.start_xytheta, self.goal_xytheta = pairs[problem_index]
        self.start_state = self.dynamics.get_state_from_xytheta(self.start_xytheta)
        self.goal_state  = self.dynamics.get_state_from_xytheta(self.goal_xytheta)

        # --- goal in grid indices (bottom-origin) ---
        row = int(np.floor(self.goal_xytheta[1] / resolution))   
        col = int(np.floor(self.goal_xytheta[0] / resolution))   

        # --- distance map on bottom-origin grid ---
        dist_arr = self._compute_distance_map(free_mask, np.array([row, col]), resolution)
        unreachable = np.isinf(dist_arr)
        reachable   = ~unreachable
        self.max_potential = float(np.max(dist_arr[reachable])) if np.any(reachable) else 0.0
        dist_arr[unreachable] = self.max_potential
        self.dist_map = dist_arr  

        # --- gradient field (unit vectors) ---
        gy_e, gx_e = np.gradient(self.dist_map, resolution, edge_order=2)
        mag = np.hypot(gx_e, gy_e)
        self.grad_x = np.zeros_like(gx_e, dtype=np.float32)
        self.grad_y = np.zeros_like(gy_e, dtype=np.float32)
        nz = mag > 1e-8
        self.grad_x[nz] = -gx_e[nz] / mag[nz]
        self.grad_y[nz] = -gy_e[nz] / mag[nz]
        valid = (free_mask == 1) & (~unreachable)
        self.grad_x[~valid] = 0.0
        self.grad_y[~valid] = 0.0

        # --- reward weights ---
        self.align_coeff       =  1.0
        self.time_cost         = -0.2
        self.completion_bonus  = 80.0
        self.collision_penalty = -10.0

        # --- action & observation spaces ---
        self.n_actions      = self.dynamics.motion_primitives.shape[0]
        self.action_space   = spaces.Discrete(self.n_actions)
        self.hist_len       = hist_len
        self.action_history = deque(maxlen=hist_len)

        # --- lidar ---
        self.n_lasers     = n_lasers
        self.laser_angles = np.linspace(-np.pi, np.pi, n_lasers, endpoint=False)

        self.observation_space = spaces.Dict({
            "scan":       spaces.Box(0.0, max_scan_dist, (n_lasers,), dtype=np.float32),
            "goal_vec":   spaces.Box(-np.inf, np.inf, (2,),       dtype=np.float32),
            "action_mask":spaces.MultiBinary(self.n_actions),
            "hist":       spaces.MultiDiscrete([self.n_actions]*hist_len),
            "dist_grad":  spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
            "dist_phi":   spaces.Box(0.0, 1.0,   (1,), dtype=np.float32),
        })

        # --- internal state ---
        self.max_steps = int(max_steps)
        self.angle_tol = float(angle_tol)
        self.state: Optional[np.ndarray] = None
        self._steps = 0
        self._fig = None
        self._ax  = None

        # constraints stored only for masking / debugging
        self.constraints: list[WorldConstraint] = []

        # --- precise time accumulator for "act" timeline ---
        self._t_acc: float = 0.0

        # debugging
        self.debug_constraints = True
        self._violations_logged = 0
        self.log_every = 1
        self.log_min_clear = None

        self.constraint_radius = 0.10   
        self.constraint_time_slack = 10.00 

    # ----------------------------------------------------------------------

    def reset(self, *, seed=None, options=None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self.state = self.start_state.copy()
        self._steps = 0
        self._t_acc = 0.0
        self.action_history.clear()
        for _ in range(self.hist_len):
            self.action_history.append(0)
        return self._get_obs(), {}

    def step(self, action: int):
        assert self.state is not None
        cur = self.state.copy()

        traj = self.dynamics.get_next_states(cur)[action, :, :3]
        valid = self.world_cc.isValid(self.footprint, traj).all()

        # per-step constraint logging (does not affect dynamics)
        self._log_constraint_check(cur, traj, action)

        if not valid:
            reward = self.collision_penalty + self.time_cost
            self.action_history.append(action)
        else:
            self.state = traj[-1]
            self.action_history.append(action)

            prev_phi = self._interpolate_dist(cur[:2])
            new_phi  = self._interpolate_dist(self.state[:2])
            reward = (prev_phi - new_phi) + self.time_cost

        # advance precise act timeline AFTER logging
        dt_action = float(self.dynamics.motion_primitives[action, -1])
        self._t_acc += dt_action

        # advance step counter
        self._steps += 1

        dxg = self.state[0] - self.goal_state[0]
        dyg = self.state[1] - self.goal_state[1]
        dtg = self.state[2] - self.goal_state[2]
        reached = valid and (np.hypot(dxg, dyg) < 0.2)
        if reached:
            reward += self.completion_bonus

        done = reached or (self._steps >= self.max_steps)
        info = {"collision": not valid, "reached": reached}
        return self._get_obs(), reward, done, False, info

    def set_constraints(self, constraints: list[WorldConstraint]):
        self.constraints = sorted(constraints, key=lambda c: c.get_end_constraint_time())

    @staticmethod
    def _seg_point_dist(p0, p1, q):
        v = p1 - p0; w = q - p0
        vv = float(np.dot(v, v))
        if vv < 1e-12:
            return float(np.linalg.norm(w))
        t = np.clip(float(np.dot(w, v) / vv), 0.0, 1.0)
        proj = p0 + t * v
        return float(np.linalg.norm(q - proj))


    def action_masks(self) -> np.ndarray:
        assert self.state is not None

        trajs = self.dynamics.get_next_states(self.state)[:, :, :3]
        mask = np.zeros(self.n_actions, dtype=bool)

        dt_default = float(self.dynamics.motion_primitives[0, -1])
        t0_mask = self._steps * dt_default

        map_ok    = np.zeros(self.n_actions, dtype=bool)
        st_ok     = np.zeros(self.n_actions, dtype=bool)
        clearance = np.full(self.n_actions, 1e9, dtype=np.float32)

        constr_pts, constr_ts = [], []
        for c in self.constraints:
            constr_pts.append(np.array(c.get_point(), dtype=np.float32))
            constr_ts.append(float(c.get_start_constraint_time()))
        constr_pts = np.asarray(constr_pts, dtype=np.float32) if len(constr_pts) > 0 else None
        constr_ts  = np.asarray(constr_ts,  dtype=np.float32) if len(constr_ts)  > 0 else None

        R   = float(self.constraint_radius)
        TSL = float(self.constraint_time_slack)
        EPS = 1e-3 

        banned_by_radius = 0
        banned_by_core   = 0

        for i in range(self.n_actions):
            poly3 = trajs[i]
            if not self.world_cc.isValid(self.footprint, poly3).all():
                continue
            map_ok[i] = True

            dt_i = float(self.dynamics.motion_primitives[i, -1])

            poses_mask = np.stack([self.state[:3], poly3[-1]], axis=0)
            times_mask = np.array([t0_mask, t0_mask + dt_i], dtype=float)
            violates_core = any(
                c.violated_constraint(poses_mask, times_mask, self.footprint, self.world_cc)
                for c in self.constraints
            )

            violates_rad = False
            min_d = float("inf")
            if constr_pts is not None:
                t0 = t0_mask - TSL
                t1 = t0_mask + dt_i + TSL
                idx = np.where((constr_ts >= t0) & (constr_ts <= t1))[0]
                if idx.size > 0:
                    poly2 = poly3[:, :2].astype(np.float32)
                    for j in idx:
                        q = constr_pts[j]
                        for k in range(len(poly2) - 1):
                            d = self._seg_point_dist(poly2[k], poly2[k+1], q)
                            if d < min_d:
                                min_d = d
                            if d < (R + EPS):
                                violates_rad = True
                    clearance[i] = min_d if np.isfinite(min_d) else clearance[i]

            violates = (violates_core or violates_rad)
            if violates:
                banned_by_radius += int(violates_rad)
                banned_by_core   += int(violates_core)
            else:
                st_ok[i] = True

        final_ok = map_ok & st_ok

        if not final_ok.any():
            cand = np.where(map_ok)[0]
            if cand.size > 0:
                top = cand[np.argsort(-clearance[cand])[:2]]
                final_ok[top] = True

        mask[:] = final_ok

        # if self.debug_constraints:
        #     print(
        #         f"[mask] step={self._steps} "
        #         f"map_ok={int(map_ok.sum())}/{self.n_actions} "
        #         f"final_ok={int(final_ok.sum())}/{self.n_actions} "
        #         f"banned_by_radius={banned_by_radius} banned_by_core={banned_by_core} "
        #         f"R={R:.3f} TSL={TSL:.3f}"
        #     )

        return mask

    # ----------------------------------------------------------------------

    def _get_obs(self) -> Dict:
        x, y, theta = self.state[:3]

        # lidar (map only)
        scans = np.full(self.n_lasers, self.max_scan_dist, dtype=np.float32)
        H, W = self.dist_map.shape
        max_steps = int(self.max_scan_dist / self.resolution)
        for idx, ang in enumerate(self.laser_angles):
            dx = np.cos(theta+ang); dy = np.sin(theta+ang)
            for s in range(max_steps):
                px = x + dx*s*self.resolution
                py = y + dy*s*self.resolution
                ix = int(np.floor(px/self.resolution))
                iy = int(np.floor(py/self.resolution))
                # print("grid shape: ", self.world_cc.grid.shape)
                # print("H, W: ", H, W)
                # print("ix, iy: ", ix, iy)
                if not (0 <= ix < W and 0 <= iy < H) or self.world_cc.grid[ix, iy] != 0:
                    scans[idx] = s * self.resolution
                    break

        # goal vector in robot frame
        dxg = self.goal_state[0]-x; dyg = self.goal_state[1]-y
        gx =  dxg*np.cos(-theta) - dyg*np.sin(-theta)
        gy =  dxg*np.sin(-theta) + dyg*np.cos(-theta)
        goal_vec = np.array([gx, gy], dtype=np.float32)

        # dist-gradient in robot frame
        gx_w = self._interpolate_grad(self.grad_x, self.state[:2])
        gy_w = self._interpolate_grad(self.grad_y, self.state[:2])
        gx_r = gx_w*np.cos(-theta) - gy_w*np.sin(-theta)
        gy_r = gx_w*np.sin(-theta) + gy_w*np.cos(-theta)

        # normalized distance
        phi = self._interpolate_dist(self.state[:2])
        phi_norm = phi / (self.max_potential + 1e-6)

        hist = np.array(self.action_history, dtype=np.int64)
        return {
            "scan":        scans,
            "goal_vec":    goal_vec,
            "action_mask": self.action_masks(),
            "hist":        hist,
            "dist_grad":   np.array([gx_r, gy_r], dtype=np.float32),
            "dist_phi":    np.array([phi_norm], dtype=np.float32),
        }

    # ----------------------------------------------------------------------

    def render(self, mode="human"):
        import matplotlib.pyplot as plt

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6,6))
        ax = self._ax
        ax.clear()

        # draw grid, start, robot
        addGridToPlot(self.world_cc, ax)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.start_xytheta)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.state[:3])

        # visualize constraints only
        for c in self.constraints:
            x, y = c.get_point()
            ax.scatter(x, y, marker='x', s=100, c='yellow', linewidths=2, zorder=6)
            circ = Circle((x, y), radius=self.constraint_radius, fill=False,
                        edgecolor='red', linewidth=1.0, alpha=0.9, zorder=6)
            ax.add_patch(circ)

        # heatmap
        H, W = self.dist_map.shape
        dm = self.dist_map / (self.max_potential + 1e-6)
        ax.imshow(
            dm,
            origin='lower',
            extent=[0, W*self.resolution, 0, H*self.resolution],
            cmap='viridis',
            alpha=0.5
        )
        ax.set_xlim(0, W*self.resolution)
        ax.set_ylim(0, H*self.resolution)
        ax.set_aspect('equal')
        ax.set_title(f"Step {self._steps}")

        if mode == "rgb_array":
            self._fig.canvas.draw()
            w, h = self._fig.canvas.get_width_height()
            buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            return buf.reshape(h, w, 3)
        else:
            plt.pause(1/self.metadata["render_fps"])

    # ----------------------------------------------------------------------

    @staticmethod
    def _compute_distance_map(grid: np.ndarray, goal_xy: np.ndarray, resolution: float) -> np.ndarray:
        from collections import deque
        H, W = grid.shape
        free = (grid == 1)
        dist = np.full((H, W), np.inf, dtype=np.float32)

        gx, gy = goal_xy.astype(int)
        # if goal on obstacle, find nearest free
        if not free[gx, gy]:
            visited = np.zeros_like(free, bool)
            dq = deque([(gx, gy)])
            visited[gx, gy] = True
            neighs = [(-1,0),(1,0),(0,-1),(0,1)]
            while dq:
                x0, y0 = dq.popleft()
                for dx, dy in neighs:
                    xn, yn = x0+dx, y0+dy
                    if 0<=xn<H and 0<=yn<W and not visited[xn,yn]:
                        if free[xn,yn]:
                            gx, gy = xn, yn
                            dq.clear()
                            break
                        visited[xn,yn] = True
                        dq.append((xn,yn))

        dist[gx, gy] = 0.0
        dq = deque([(gx, gy)])
        neighs = [(-1,0),(1,0),(0,-1),(0,1)]
        while dq:
            x0, y0 = dq.popleft()
            for dx, dy in neighs:
                xn, yn = x0+dx, y0+dy
                if (0<=xn<H and 0<=yn<W and free[xn,yn]
                        and not np.isfinite(dist[xn,yn])):
                    dist[xn,yn] = dist[x0,y0] + 1.0
                    dq.append((xn,yn))
        return dist

    def _interpolate_dist(self, xy: np.ndarray) -> float:
        x, y = xy
        i_f = np.clip(y/self.resolution, 0, self.dist_map.shape[0]-1)
        j_f = np.clip(x/self.resolution, 0, self.dist_map.shape[1]-1)
        i0, j0 = int(np.floor(i_f)), int(np.floor(j_f))
        i1, j1 = min(i0+1, self.dist_map.shape[0]-1), min(j0+1, self.dist_map.shape[1]-1)
        di, dj = i_f - i0, j_f - j0
        d00 = self.dist_map[i0, j0]; d10 = self.dist_map[i1, j0]
        d01 = self.dist_map[i0, j1]; d11 = self.dist_map[i1, j1]
        return (
            d00*(1-di)*(1-dj)
          + d10*di*(1-dj)
          + d01*(1-di)*dj
          + d11*di*dj
        )

    def _interpolate_grad(self, grad_map: np.ndarray, xy: np.ndarray) -> float:
        x, y = xy
        i_f = np.clip(y/self.resolution, 0, self.dist_map.shape[0]-1)
        j_f = np.clip(x/self.resolution, 0, self.dist_map.shape[1]-1)
        i0, j0 = int(np.floor(i_f)), int(np.floor(j_f))
        i1, j1 = min(i0+1, grad_map.shape[0]-1), min(j0+1, grad_map.shape[1]-1)
        di, dj = i_f - i0, j_f - j0
        g00 = grad_map[i0, j0]; g10 = grad_map[i1, j0]
        g01 = grad_map[i0, j1]; g11 = grad_map[i1, j1]
        return (
            g00*(1-di)*(1-dj)
          + g10*di*(1-dj)
          + g01*(1-di)*dj
          + g11*di*dj
        )

    def _log_constraint_check(self, cur_state, traj, action_idx):
        """
        Logging only: check this executed action against constraints
        on BOTH timelines:
        - mask timeline:  t0_mask = _steps * dt_default
        - act timeline:   t0_act  = self._t_acc (accumulated real time)
        Prints every `self.log_every` steps, or whenever min_clear <= self.log_min_clear,
        and always prints on hits.
        """
        if not self.debug_constraints or len(self.constraints) == 0:
            return

        dt_default = float(self.dynamics.motion_primitives[0, -1])
        dt_action  = float(self.dynamics.motion_primitives[action_idx, -1])

        # timelines
        t0_mask = self._steps * dt_default
        t1_mask = t0_mask + dt_action

        t0_act  = self._t_acc
        t1_act  = t0_act + dt_action

        poses = np.stack([cur_state[:3], traj[-1]], axis=0)
        times_mask = np.array([t0_mask, t1_mask], dtype=float)
        times_act  = np.array([t0_act,  t1_act ], dtype=float)

        v_mask = any(
            c.violated_constraint(poses, times_mask, self.footprint, self.world_cc)
            for c in self.constraints
        )
        v_act  = any(
            c.violated_constraint(poses, times_act,  self.footprint, self.world_cc)
            for c in self.constraints
        )

        # spatial nearest distance (visual aid)
        def seg_point_dist(p0, p1, q):
            v = p1 - p0; w = q - p0
            vv = float(np.dot(v, v))
            if vv < 1e-12:
                return float(np.linalg.norm(w))
            t = np.clip(float(np.dot(w, v) / vv), 0.0, 1.0)
            proj = p0 + t * v
            return float(np.linalg.norm(q - proj))

        min_clear = float("inf"); near_xy = (np.nan, np.nan)
        p0 = cur_state[:2].astype(np.float32)
        p1 = traj[-1, :2].astype(np.float32)
        for c in self.constraints:
            q = np.array(c.get_point(), dtype=np.float32)
            d = seg_point_dist(p0, p1, q)
            if d < min_clear:
                min_clear = d
                near_xy = (float(q[0]), float(q[1]))

        # Always print on hits
        if v_mask or v_act:
            self._violations_logged += 1
            # print(
            #     f"[CONSTR-HIT] step={self._steps} act={action_idx}  "
            #     f"mask=[{t0_mask:.3f},{t1_mask:.3f}] -> {v_mask}  "
            #     f"act=[{t0_act:.3f},{t1_act:.3f}]   -> {v_act}  "
            #     f"min_clear={min_clear:.3f}  near=({near_xy[0]:.3f},{near_xy[1]:.3f})"
            # )
            return

        # Otherwise: print every step (log_every=1), or when clearance under threshold
        need_print = (self._steps % max(1, int(self.log_every)) == 0)
        if (self.log_min_clear is not None) and (min_clear <= float(self.log_min_clear)):
            need_print = True

        # if need_print:
        #     print(
        #         f"[constr-check] step={self._steps} act={action_idx}  "
        #         f"mask=[{t0_mask:.3f},{t1_mask:.3f}]  act=[{t0_act:.3f},{t1_act:.3f}]  "
        #         f"min_clear={min_clear:.3f}"
        #     )