# evaluate.py

from stable_baselines3 import PPO
from potato_tower_env import PotatoTowerEnv
import time

# Load trained PPO model
model = PPO.load("ppo_potato_tower")

# Create environment
env = PotatoTowerEnv()

# Reset environment
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward

    # Print live debug info
    print(f"Step: {env.current_index}, Reward: {reward:.2f}, Total: {total_reward:.2f}")

    time.sleep(0.2)  # slow it down so you can follow (can remove later)

print("\n✅ Evaluation complete.")
print(f"🎯 Final tower height (total reward): {total_reward:.2f}")
