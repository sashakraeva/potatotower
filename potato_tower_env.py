import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymunk
from pymunk import Body, Poly
import pygame

from potato_shapes import load_all_potatoes


class PotatoTowerEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.pixels_per_meter = 1.0

        self.width = 800
        self.height = 600
        self.max_potatoes = 10
        self.gravity = -900
        self.floor_height = 50

        self.potato_polygons = load_all_potatoes("svg")
        self.num_potatoes = len(self.potato_polygons)

        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -np.pi, 0]),
            high=np.array([self.width, np.pi, self.num_potatoes - 1]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.max_potatoes * 5 + 1,),
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
        self.remaining_potato_indices = list(range(self.num_potatoes))

    def _add_ground(self):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Segment(body, (0, self.floor_height), (self.width, self.floor_height), 10)
        shape.friction = 2.0
        self.space.add(body, shape)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup()
        return self._get_obs(), {}

    def step(self, action):
        if self.current_index >= self.max_potatoes:
            return self._get_obs(), 0.0, True, False, {}

        x_pos, angle, potato_index = action
        x_pos = float(np.clip(x_pos, 0, self.width))
        angle = float(np.clip(angle, -np.pi, np.pi))
        potato_index = int(np.clip(round(potato_index), 0, self.num_potatoes - 1))

        if potato_index not in self.remaining_potato_indices:
            reward = -1000
            done = True
            return self._get_obs(), reward, done, False, {}

        self.remaining_potato_indices.remove(potato_index)
        poly_points = self.potato_polygons[potato_index]

        mass = 1.0
        moment = pymunk.moment_for_poly(mass, poly_points)
        body = Body(mass, moment)
        body.position = (x_pos, self.height - 100)
        body.angle = angle
        shape = Poly(body, poly_points)
        shape.friction = 1.0

        self.space.add(body, shape)
        self.bodies.append(body)

        x, y = body.position
        self.placed_potatoes.append((x, y, angle, self._poly_width(poly_points), self._poly_height(poly_points)))
        self.current_index += 1

        for _ in range(30):
            self.space.step(1 / 60.0)

        done = self.current_index >= self.max_potatoes or self._tower_collapsed()
        reward = self._compute_reward() if not done else 0.0

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        obs = []
        for (x, y, angle, w, h) in self.placed_potatoes:
            obs.extend([x, y, angle, w, h])
        obs += [0.0] * (5 * (self.max_potatoes - len(self.placed_potatoes)))
        obs.append(self.current_index / self.max_potatoes)
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self):
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

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Potato Tower")
            self.clock = pygame.time.Clock()

        self.screen.fill((240, 240, 240))

        for i, shape in enumerate(self.space.shapes):
            if isinstance(shape, pymunk.Poly):
                body = shape.body
                points = [body.local_to_world(v) for v in shape.get_vertices()]
                screen_points = [
                    (int(p[0]), int(self.height - p[1]))
                    for p in points
                ]
                color = (
                    150 + (i * 17) % 100,
                    80 + (i * 31) % 50,
                    40 + (i * 11) % 60
                )
                pygame.draw.polygon(self.screen, color, screen_points)

        pygame.display.flip()
        self.clock.tick(30)