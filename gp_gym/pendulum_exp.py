from pendulum_top_quadrants import Pendulum
from gp_gym_info import info
import numpy as np
import matplotlib.pyplot as plt


# --- Parameters --- #
# Experiment Parameters
num_runs = 10

# Environment Parameters
info["env_name"] = "Pendulum-v0"

# GP Parameters
info["pop_size"] = 200
info["max_gens"] = 20
info["term_fit"] = -200
info["tournament_size"] = 10
info["mutation_rate"] = 0.1
info["max_depth"] = 5

# Fitness Evaluation (training)
info["num_eps"] = 4
info["num_time_steps"] = 500

# Fitness evaluation (testing)
num_eps_test = 100
num_steps_test = 200



# --- Run experiment --- #
avg_fitnesses = np.zeros((num_runs, info["max_gens"]))
best_programs = []
best_program_fitnesses = np.zeros(num_runs)
for i in range(num_runs):

    print("\n----- Run {} -----".format(i+1))

    # Train agent
    agent = Pendulum(info)
    best_program = agent.train()

    # Save average generation fitnesses
    avg = agent.logbook.select("avg")
    avg_fitnesses[i] = avg

    # Evaluate best program of this run
    fitness = agent.fit(best_program, num_eps_test, num_steps_test)[0]
    
    # Save the best program & its fitness
    best_programs.append(best_program)
    best_program_fitnesses[i] = fitness



# --- Show Results --- #
# Compute average fitness of each generation over all of the runs of the experiment
# i.e. [avg_gen1_fit, avg_gen2_fit, ..., avg_gen_n_fit]
avg_fit = np.mean(avg_fitnesses, axis=0)
gens = agent.logbook.select("gen")  # [1, 2, ..., max_gens]

# Plot average fitness vs generations
plt.plot(gens, avg_fit)
plt.xlabel("generation")
plt.ylabel("average fitness")
plt.xticks(ticks=gens)
plt.show()

# Print average solution fitness over all runs of the experiment
mean_solution_fit = np.mean(best_program_fitnesses)
print("\nAverage solution fitness over {} runs of the experiment: {}".format(num_runs, mean_solution_fit))

# Print best programs
print("\nBest programs:")
for p in best_programs:
    print(p)
