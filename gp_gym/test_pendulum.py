from cartpole import CartPole
from cartpole_info import info
from gp_gym import gen_program, gen_init_pop, run_ep_while_not_done
import gym
from numpy import random


random.seed(0)

T = info["T"]
F = info["F"]
max_depth = 1
method = "grow"
t_rate = 1
p_type = "Action"


pop = gen_init_pop(10, T, F, max_depth, method, t_rate, p_type)
cartpole_agent = CartPole(info)
scores = cartpole_agent.batch_fit(pop, 100)

print("Fitness scores: {}".format(scores))
