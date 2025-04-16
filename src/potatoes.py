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
        self.max_potatoes = 5  # Number of unique potatoes allowed
        self.angle_bins = 12   # How many rotation bins (e.g., 30° increments)
        self.svg_dir = svg_dir

        # Action space: [potato index, rotation angle]
        self.action_space = spaces.MultiDiscrete([self.max_potatoes, self.angle_bins])

        # Observation: used_flags + (x, y, angle) for each potato + tower height
        low = np.array([0] * self.max_potatoes + [0, 0, 0] * self.max_potatoes + [0], dtype=np.float32)
        high = np.array([1] * self.max_potatoes + [self.width, self.height, 360] * self.max_potatoes + [self.height], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.space = None
        self.bodies = []
        self.potato_shapes = self._load_svg_polygons()

        # Setup for Pygame rendering if enabled
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.font = pygame.font.SysFont("Arial", 24)

    def _load_svg_polygons(self):
        # Convert SVG paths to shapely polygons, scaled and centered
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
                poly = Polygon([(0, 0), (30, 0), (15, 40)])  # Fallback triangle
                polys.append(poly)
        self.max_potatoes = len(polys)
        return polys

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.space = pymunk.Space()
        self.space.gravity = (0, 500)
        self.bodies = []
        self.used_flags = [0] * self.max_potatoes
        self.positions = [[0, 0, 0] for _ in range(self.max_potatoes)]
        self.tower_height = 0
        self.step_count = 0
        self.episode_id = getattr(self, "episode_id", 0)  # fallback to 0 if not set externally

        # Ground plane
        ground = pymunk.Segment(self.space.static_body, (0, self.ground_y), (self.width, self.ground_y), 5)
        ground.friction = 1.0
        self.space.add(ground)

        return self._get_obs(), {}

    def step(self, action):
        index, angle_idx = action
        angle = angle_idx * (360 / self.angle_bins)
        print(f"Action chosen: potato #{index}, angle {angle:.1f}°")
        self.step_count += 1

        if self.used_flags[index]:
            print(f"❌ Potato {index} already used!")
            return self._get_obs(), -200, self.step_count >= self.max_potatoes, False, {}

        # Rotate and transform potato shape
        poly = self.potato_shapes[index]
        rotated = affinity.rotate(poly, angle, origin='center')
        coords = [(float(p[0]), float(p[1])) for p in rotated.exterior.coords]
        if len(coords) < 3:
            return self._get_obs(), -10, False, False, {}

        # Placement
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

        self._update_tower_height()
        reward = self.tower_height * 2.0

        # ─── Centering ───
        center_x = self.width / 2
        dx_from_center = abs(body.position.x - center_x)
        reward += max(0, 10 - dx_from_center) if dx_from_center <= 30 else -10

        # ─── Contacts ───
        contacts = sum(1 for arbiter in self.space.shape_query(shape) if arbiter.shape != shape)
        if len(self.bodies) == 1 and body.position.y < self.ground_y - 60:
            reward += 10
        elif contacts == 1:
            reward += 20
        elif contacts == 2 and 1 < len(self.bodies) < self.max_potatoes:
            reward += 40
        else:
            reward -= 20

        # ─── Alignment ───
        dx, dy = 0.0, 0.0  # default if it's the first potato
        if len(self.bodies) > 1:
            new_body = self.bodies[-1][0]
            prev_body = self.bodies[-2][0]
            dx = abs(new_body.position.x - prev_body.position.x)
            dy = new_body.position.y - prev_body.position.y
            if dy < 10 or dx > 15:
                print(f"⚠️ Not stacked well: dx: {dx:.2f}, dy: {dy:.2f}")
                reward -= 10 + 0.2 * dx + 0.2 * abs(dy)


        # ─── Reward for stacking with unique Y values ───
        y_vals = [round(body.position.y, 1) for body, _ in self.bodies]
        unique_y = len(set(y_vals))

        reward += (unique_y - 1) * 10  # e.g. 2 unique Y → +10, 3 → +20...

        # ─── Additional reward for same X alignment ───
        x_vals = [round(body.position.x, 1) for body, _ in self.bodies]
        x_centered = sum(1 for x in x_vals if abs(x - center_x) < 10)

        if x_centered >= 2 and unique_y >= x_centered:
            reward += (x_centered - 1) * 10
            print(f"📏 Column bonus: {x_centered} potatoes aligned in X with {unique_y} unique Y → +{(x_centered - 1) * 10}")

        # ─── Perfect Stack Bonus ───
        if all(self.used_flags):
            x_spread = max(x_vals) - min(x_vals)
            if x_spread <= 10 and unique_y == self.max_potatoes:
                reward += 150
                print("🏆 Perfect column stack achieved!")

        # ─── Endgame Bonuses ───
        if all(self.used_flags):
            reward += 15
        if len(self.bodies) == self.max_potatoes:
            grounded = sum(1 for body, _ in self.bodies if body.position.y > self.ground_y - 5)
            if grounded == 1:
                reward += 50

        done = all(self.used_flags) or self.step_count >= self.max_potatoes

       # ─── LOG TRAINING STEP TO CSV ───
        log_path = "training_log.csv"
        write_header = not os.path.isfile(log_path) or os.stat(log_path).st_size == 0

        with open(log_path, "a") as f:
            if write_header:
                f.write("episode,step,potato_index,angle,x,y,reward,tower_height,contacts,dx,dy,potatoes_used\n")
            f.write(
                f"{self.episode_id},"                    # Episode number
                f"{self.step_count},"                    # Step number in episode
                f"{index},"                              # Potato index used
                f"{angle:.1f},"                          # Angle in degrees
                f"{x:.2f},"                              # X position
                f"{y:.2f},"                              # Y position
                f"{reward:.2f},"                         # Reward
                f"{self.tower_height:.2f},"              # Tower height
                f"{contacts},"                           # Contacts
                f"{dx:.2f},"                             # dx
                f"{dy:.2f},"                             # dy
                f"{sum(self.used_flags)}\n"              # Potatoes placed so far
            )


        return self._get_obs(), reward, done, False, {}
    







    def _get_obs(self):
        norm_positions = [[x / self.width, y / self.height, angle / 360] for x, y, angle in self.positions]
        flat_positions = [coord for pos in norm_positions for coord in pos]
        norm_tower = self.tower_height / self.height
        return np.array(self.used_flags + flat_positions + [norm_tower], dtype=np.float32)


    def _update_tower_height(self):
        # Calculate the distance from ground to top of the topmost potato
        if not self.bodies:
            self.tower_height = 0
            return
        top_y = max(body.position.y for body, _ in self.bodies)
        self.tower_height = self.ground_y - top_y

    def render(self):
        if self.render_mode != "human":
            return
        self.screen.fill((255, 255, 255))

        # Draw physics world (potatoes + ground)
        self.space.debug_draw(self.draw_options)

        # Draw tower height in top-left corner
        text = self.font.render(f"Height: {self.tower_height:.2f}", True, (0, 0, 0))
        self.screen.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
