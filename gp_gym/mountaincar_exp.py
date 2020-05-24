from mountaincar import MountainCar
from gp_gym_info import info
import gym

from deap import gp

import random
random.seed(5)

# env = gym.make("MountainCarContinuous-v0")
# print(env.action_space.low)

# GP Parameters
info["env_name"] = "MountainCarContinuous-v0"
info["pop_size"] = 100
info["max_gens"] = 10
info["max_depth"] = 1
info["tournament_size"] = 5
info["num_eps"] = 10

agent = MountainCar(info)
best_program = "IFLTE(0.0, velocity, 0.18, -0.18)"
print(best_program)
f = agent.fit(best_program, 100, 200, render=False)
print(f)
