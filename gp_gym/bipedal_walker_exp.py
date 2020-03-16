from bipedal_walker import BipedalWalker
from gp_gym_info import info
import gym
import numpy as np

np.random.seed(1)

info["env_name"] = "BipedalWalker-v3"
info["pop_size"] = 10
info["max_gens"] = 5
info["num_eps"]  = 10
info["num_time_steps"]  = 100
info["tournament_size"] = 3
info["mutation_rate"] = 0.1
info["term_fit"] = 300
info["max_depth"] = 5

agent = BipedalWalker(info)
best_program = agent.train()
print(best_program)
fitness = agent.fit(best_program, info["num_eps"], info["num_time_steps"], render=False)
print("Fitness: {}".format(fitness))

agent.fit(best_program, 5, 100, render=True)
