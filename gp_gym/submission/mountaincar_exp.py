from mountaincar import MountainCar
from gp_gym_info import info
import gym


# GP Parameters
info["env_name"] = "MountainCar-v0"
info["pop_size"] = 100
info["max_gens"] = 10
info["max_depth"] = 1
info["num_eps"] = 100

agent = MountainCar(info)
best_program = agent.train()
print(best_program)
f = agent.fit(best_program, 100, 200, render=False)
print(f)
