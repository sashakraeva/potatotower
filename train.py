
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from potato_tower_env import PotatoTowerEnv

# Create environment instance
env = PotatoTowerEnv()

# Optional: check for Gym compatibility issues
check_env(env)

# Create PPO model with a multilayer perceptron (MLP) policy
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100_000)

# Save the trained model
model.save("ppo_potato_tower")
