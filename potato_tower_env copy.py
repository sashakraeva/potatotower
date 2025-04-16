import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymunk
from pymunk import Body, Poly

from potato_shapes import load_all_potatoes


class PotatoTowerEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.width = 800
        self.height = 600
        self.max_potatoes = 10
        self.gravity = -900
        self.floor_height = 50

        self.space = pymunk.Space()
        self.space.gravity = (0, self.gravity)

        # Load all potato shapes
        self.potato_polygons = load_all_potatoes("svg")

        # Action: [x position (0-800), rotation angle (-pi to pi)]
        self.action_space = spaces.Box(
            low=np.array([0.0, -np.pi]),
            high=np.array([self.width, np.pi]),
            dtype=np.float32
        )

        # Observation: all placed potatoes + current shape size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.max_potatoes * 5 + 1,),  # x, y, angle, w, h for each + index
            dtype=np.float32
        )

        self._setup()

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, self.gravity)
        self._add_ground()
        self.placed_potatoes = []
        self.bodies = []
        self.current_index = 0

    def _add_ground(self):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Segment(body, (0, self.floor_height), (self.width, self.floor_height), 5)
        shape.friction = 1.0
        self.space.add(body, shape)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup()
        return self._get_obs(), {}

    def step(self, action):
        if self.current_index >= len(self.potato_polygons):
            return self._get_obs(), 0.0, True, False, {}

        x_pos, angle = action
        x_pos = float(np.clip(x_pos, 0, self.width))
        angle = float(np.clip(angle, -np.pi, np.pi))

        poly_points = self.potato_polygons[self.current_index]
        mass = 1.0
        moment = pymunk.moment_for_poly(mass, poly_points)
        body = Body(mass, moment)
        body.position = (x_pos, self.height - 50)  # drop from top
        body.angle = angle
        shape = Poly(body, poly_points)
        shape.friction = 1.0

        self.space.add(body, shape)
        self.bodies.append(body)

        # Store for observation
        x, y = body.position
        self.placed_potatoes.append((x, y, angle, self._poly_width(poly_points), self._poly_height(poly_points)))
        self.current_index += 1

        # Simulate physics
        for _ in range(30):
            self.space.step(1 / 60.0)

        done = self.current_index >= self.max_potatoes or self._tower_collapsed()
        reward = self._compute_reward() if not done else 0.0

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        obs = []
        for (x, y, angle, w, h) in self.placed_potatoes:
            obs.extend([x, y, angle, w, h])
        # Pad the rest
        obs += [0.0] * (5 * (self.max_potatoes - len(self.placed_potatoes)))
        obs.append(self.current_index / self.max_potatoes)  # normalized progress
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self):
        # Reward = height of the highest potato
        heights = [b.position.y for b in self.bodies]
        return max(heights, default=0.0)

    def _tower_collapsed(self):
        for b in self.bodies:
            if b.position.y < self.floor_height or abs(b.angle) > np.pi / 2:
                return True
        return False

    def _poly_width(self, poly):
        xs = [p[0] for p in poly]
        return max(xs) - min(xs)

    def _poly_height(self, poly):
        ys = [p[1] for p in poly]
        return max(ys) - min(ys)
