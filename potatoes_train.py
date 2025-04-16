from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from all import PotatoTowerEnv

# Wrap the environment for vectorized training
env = DummyVecEnv([lambda: PotatoTowerEnv(render_mode=None)])

# Create and train the PPO model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# Save the trained model
model.save("ppo_potato")
