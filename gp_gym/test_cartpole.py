from cartpole import CartPole
from cartpole_info import info
from gp_gym import gen_program
from numpy import random


random.seed(0)

T = info["T"]
F = info["F"]
max_depth = 3
method = "grow"
t_rate = 0.5
p_type = "Action"

p1 = gen_program(T, F, max_depth, method, t_rate, p_type)
print("p1: {}".format(p1))

obs = [1.0, 2.0, 3.0, 4.0]

cartpole_agent = CartPole(info)
action = cartpole_agent.eval(p1, obs)
print("Action: {}".format(action))
