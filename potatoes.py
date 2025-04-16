import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import pymunk
import pymunk.pygame_util
import pygame
from svgpathtools import svg2paths
from shapely.geometry import Polygon
from shapely import affinity

class PotatoTowerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, svg_dir="svg_01"):
        super().__init__()
        self.render_mode = render_mode
        self.width = 800
        self.height = 600
        self.ground_y = 550
        self.max_potatoes = 5
        self.angle_bins = 12
        self.svg_dir = svg_dir

        self.action_space = spaces.MultiDiscrete([self.max_potatoes, self.angle_bins])
        low = np.array([0] * self.max_potatoes + [0, 0, 0] * self.max_potatoes + [0], dtype=np.float32)
        high = np.array([1] * self.max_potatoes + [self.width, self.height, 360] * self.max_potatoes + [self.height], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.space = None
        self.bodies = []
        self.potato_shapes = self._load_svg_polygons()

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def _load_svg_polygons(self):
        polys = []
        svg_files = sorted([f for f in os.listdir(self.svg_dir) if f.endswith(".svg")])[:self.max_potatoes]
        for filename in svg_files:
            filepath = os.path.join(self.svg_dir, filename)
            try:
                paths, _ = svg2paths(filepath)
                points = []
                for path in paths:
                    for segment in path:
                        points.append((segment.start.real, segment.start.imag))
                if len(points) < 3:
                    raise ValueError("Too few points")
                poly = Polygon(points)
                poly = affinity.translate(poly, xoff=-poly.centroid.x, yoff=-poly.centroid.y)
                poly = affinity.scale(poly, xfact=0.2, yfact=0.2, origin=(0, 0))
                polys.append(poly)
            except Exception:
                poly = Polygon([(0, 0), (30, 0), (15, 40)])
                polys.append(poly)
        self.max_potatoes = len(polys)
        return polys

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.space = pymunk.Space()
        self.space.gravity = (0, 900)
        self.bodies = []
        self.used_flags = [0] * self.max_potatoes
        self.positions = [[0, 0, 0] for _ in range(self.max_potatoes)]
        self.tower_height = 0

        ground = pymunk.Segment(self.space.static_body, (0, self.ground_y), (self.width, self.ground_y), 5)
        ground.friction = 1.0
        self.space.add(ground)

        return self._get_obs(), {}

    def step(self, action):
        index, angle_idx = action
        if self.used_flags[index]:
            return self._get_obs(), -10, False, False, {}

        angle = angle_idx * (360 / self.angle_bins)
        poly = self.potato_shapes[index]
        rotated = affinity.rotate(poly, angle, origin='center')
        coords = [(float(p[0]), float(p[1])) for p in rotated.exterior.coords]
        if len(coords) < 3:
            return self._get_obs(), -5, False, False, {}

        if len(self.bodies) == 0:
            x = self.width // 2
            y = self.ground_y - 50
        else:
            prev_body = self.bodies[-1][0]
            x = prev_body.position.x
            y = prev_body.position.y - 50

        body = pymunk.Body(1, pymunk.moment_for_poly(1, coords))
        body.position = (x, y)
        shape = pymunk.Poly(body, coords)
        shape.friction = 1.0
        self.space.add(body, shape)

        self.bodies.append((body, shape))
        self.used_flags[index] = 1
        self.positions[index] = [x, y, angle]

        for _ in range(240):
            self.space.step(1 / self.metadata["render_fps"])

        if len(self.bodies) > 1:
            new_body = self.bodies[-1][0]
            prev_body = self.bodies[-2][0]
            dx = abs(new_body.position.x - prev_body.position.x)
            dy = new_body.position.y - prev_body.position.y

            if dy < 10 or dx > 20:
                print("Invalid stacking: not directly on top of previous.")
                return self._get_obs(), -100, True, False, {}

        self._update_tower_height()
        reward = self.tower_height
        done = all(self.used_flags)

        print(f"🥔 Step {sum(self.used_flags)} → Tower height: {self.tower_height:.2f}, Reward: {reward:.2f}")
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        flat_positions = [coord for pos in self.positions for coord in pos]
        return np.array(self.used_flags + flat_positions + [self.tower_height], dtype=np.float32)

    def _update_tower_height(self):
        if not self.bodies:
            self.tower_height = 0
            return
        top_y = max(body.position.y for body, _ in self.bodies)
        self.tower_height = self.ground_y - top_y

    def render(self):
        if self.render_mode != "human":
            return
        self.screen.fill((255, 255, 255))
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
