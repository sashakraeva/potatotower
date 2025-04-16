from stable_baselines3 import PPO
from potatoes import PotatoTowerEnv
import time

# Use single non-wrapped env to allow Pygame rendering
env = PotatoTowerEnv(render_mode="human")

# Write CSV header once
open("training_log.csv", "w").close()


model = PPO("MlpPolicy", env, verbose=1)

TIMESTEPS = 50_000
EPISODES = 300

for episode in range(EPISODES):
    print(f"\n Episode {episode + 1}")
    env.episode_id = episode + 1
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        env.render()
        time.sleep(0.05)

model.save("models/ppo_potato_01")
env.close()
