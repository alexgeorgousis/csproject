from pendulum_info import info
from pendulum import Pendulum
import gym
from numpy import random

random.seed(0)


"""Experiment parameters"""
info["pop_size"] = 100
info["max_gens"] = 20
info["max_depth"] = 2
info["mutation_rate"] = 0.1
info["num_eps"]  = 5 # for training
info["num_time_steps"] = 400
info["term_fit"] = -500


"""Run Experiment"""

# Train Agent
agent = Pendulum(info)
best_program = agent.train()
print("\nBest program:\n{}".format(best_program))
print("\n{}".format(agent.logbook))

# Evaluate solution
num_eps = 100
fitness = agent.fit(best_program, num_eps)[0]
print("\nFitness: {}".format(fitness))
