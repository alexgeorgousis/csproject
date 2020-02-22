from pendulum import Pendulum
from pendulum_info import info


# Experiment Parameters
num_runs = 10

# GP Parameters
info["pop_size"] = 100
info["max_gens"] = 10
info["tournament_size"] = 10
info["mutation_rate"] = 0.1
info["max_depth"] = 3

# Fitness Evaluation (training)
info["num_eps"] = 4
info["num_time_steps"] = 500

agent = Pendulum(info)
agent.train()

# Fitness Evaluation (testing)
info["num_eps"] = 100
info["num_time_steps"] = 200
