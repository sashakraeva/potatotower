import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load your data
df = pd.read_csv("training_log.csv")
grouped = df.groupby("episode")[["reward", "tower_height"]].max().reset_index()

# Plot
sns.lineplot(data=grouped, x="episode", y="reward", label="Max Reward")
sns.lineplot(data=grouped, x="episode", y="tower_height", label="Max Tower Height")

plt.title("Max Reward and Tower Height per Episode")
plt.xlabel("Episode")
plt.ylabel("Value")
plt.legend()
plt.show()
