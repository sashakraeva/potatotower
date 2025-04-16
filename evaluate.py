# evaluate.py

from stable_baselines3 import PPO
from potato_tower_env import PotatoTowerEnv
import time

model = PPO.load("ppo_potato_tower")
env = PotatoTowerEnv(render_mode="human")

obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward

    env.render()  # <== ADD THIS

print("\n✅ Evaluation complete.")
print(f"🎯 Final tower height (total reward): {total_reward:.2f}")

# Wait before closing window
time.sleep(10)
