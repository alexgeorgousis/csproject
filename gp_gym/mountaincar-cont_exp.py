from mountaincar import MountainCar
from gp_gym_info import info
from matplotlib import pyplot as plt
import gym
import numpy as np

from deap import gp

import random
random.seed(5)

# env = gym.make("MountainCarContinuous-v0")
# print(env.action_space.low)

info["env_name"] = "MountainCarContinuous-v0"
info["pop_size"] = 100
info["max_gens"] = 10
info["max_depth"] = 1
info["tournament_size"] = 5
info["num_eps"] = 10
agent = MountainCar(info)

solutions = {}
force_values = np.arange(0.0, 1.0, 0.1)
fitness_scores = []
counter = 1
for force in force_values:
    solution = "IFLTE(0.0, velocity, {}, {})".format(force, -force)
    f = agent.fit(solution, 100, 200, render=False)[0]
    fitness_scores.append(f)

    # Timing
    print(counter)
    counter += 1

    if f >= 90:
        solutions[solution] = f

# Print solutions and their fitness scores
print()
for s, f in solutions.items():
    print("{}: {}".format(s, f))

# Plot fitness scores
plt.scatter(force_values, fitness_scores)
plt.show()

plt.style.use('ggplot')
plt.bar(force_values[0:len(force_values):4], fitness_scores[0:len(fitness_scores):4])
plt.xticks(range(len(force_values[0:len(force_values):4])), force_values[0:len(force_values):4])
plt.show()

plt.style.use('ggplot')
x = ['Nuclear', 'Hydro', 'Gas', 'Oil', 'Coal', 'Biofuel']
energy = [5, 6, 15, 22, 24, 8]
x_pos = [i for i, _ in enumerate(x)]
print("x_pos: {}".format(x_pos))
plt.bar(x_pos, energy, color='green')
plt.xlabel("Energy Source")
plt.ylabel("Energy Output (GJ)")
plt.title("Energy output from various fuel sources")
plt.xticks(x_pos, x)
plt.show()
