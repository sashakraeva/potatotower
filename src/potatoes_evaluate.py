from stable_baselines3 import PPO
from potatoes import PotatoTowerEnv
import time

# Load trained model
model = PPO.load("models/ppo_potato_01")  # Adjust path if needed

# Create env with rendering enabled
env = PotatoTowerEnv(render_mode="human")

episodes = 10
best_reward = -float("inf")
best_actions = []

for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    actions = []

    print(f"\n🎬 Episode {ep+1}")

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        actions.append(action)
        total_reward += reward
        env.render()
        time.sleep(0.1)

    print(f"🏁 Episode {ep+1} finished — Total reward: {total_reward:.2f}, Tower height: {env.tower_height:.2f}")

    if total_reward > best_reward:
        best_reward = total_reward
        best_actions = actions.copy()

# Replay best episode
print("\n🎥 Replaying best sequence...")
obs, _ = env.reset()
for action in best_actions:
    obs, _, done, _, _ = env.step(action)
    env.render()
    time.sleep(0.2)

env.close()

print(f"\n🏆 Best reward: {best_reward:.2f} with Tower height: {env.tower_height:.2f}")
