from cartpole_binary_agent import Agent
import numpy as np
from matplotlib import pyplot as plt

agent = Agent()
agent.train()

# Plot average fitness scores from every generation
gens = agent.gens
scores = [[indiv[1] for indiv in pop] for pop in gens]
avg_scores = [np.mean(pop) for pop in scores]
plt.plot([i+1 for i in range(len(avg_scores))], avg_scores)
plt.show()

agent.run()
