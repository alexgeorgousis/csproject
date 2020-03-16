import cProfile
from bipedal_walker import BipedalWalker
from gp_gym_info import info


info["env_name"] = "BipedalWalker-v3"
info["pop_size"] = 100
info["max_gens"] = 10
info["num_eps"]  = 1
info["num_time_steps"]  = 1
info["tournament_size"] = 3
info["mutation_rate"] = 0.1
info["term_fit"] = 300
info["max_depth"] = 5

agent = BipedalWalker(info)
cProfile.run("agent.train()", "bipedal_walker.profile")
