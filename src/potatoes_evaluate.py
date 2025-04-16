from stable_baselines3 import PPO
from potatoes import PotatoTowerEnv
import time

# Load the trained PPO model
model = PPO.load("ppo_potato")

# Create the environment with rendering enabled
env = PotatoTowerEnv(render_mode="human")

best_reward = -float("inf")
best_obs_sequence = []
best_actions = []

# Try multiple rollouts to find the best stacking
for episode in range(20):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    obs_sequence = []
    actions = []

    print(f"\n🎬 Episode {episode + 1}")

    while not done:
        action, _ = model.predict(obs)
        obs_sequence.append(obs)
        actions.append(action)
        obs, reward, done, _, _ = env.step(action)
        total_reward = reward
        print(f"Step: Reward={reward:.2f}, Tower height={env.tower_height:.2f}")

    print(f"✅ Episode {episode + 1} ended. Total reward: {total_reward:.2f}, Tower height: {env.tower_height:.2f}")

    if total_reward > best_reward:
        best_reward = total_reward
        best_obs_sequence = obs_sequence
        best_actions = actions

# Replay best sequence
print("\n🎥 Replaying best stacking...")
obs, _ = env.reset()
for action in best_actions:
    obs, reward, done, _, _ = env.step(action)
    env.render()
    print(f"Replaying Step: Reward={reward:.2f}, Tower height={env.tower_height:.2f}")
    time.sleep(0.5)

env.close()
print("\n🏆 Final (best) tower height:", best_reward)
