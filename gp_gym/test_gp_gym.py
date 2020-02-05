from gp_gym import gen_init_pop, gen_program
from cartpole_info import info as cartpole_info
from pendulum_info import info as pendulum_info
import numpy as np


np.random.seed(0)

cartpole_info["term_growth_rate"] = .5
cartpole_info["max_depth"] = 2
cartpole_info["pop_size"] = 2

pop_size = cartpole_info["pop_size"]
T = cartpole_info["T"]
F = cartpole_info["F"]
max_depth = cartpole_info["max_depth"]
method = cartpole_info["method"]
t_rate = cartpole_info["term_growth_rate"]
p_type = cartpole_info["program_type"]

p1 = gen_program(T, F, max_depth, method, t_rate, p_type)
p2 = gen_program(T, F, max_depth, method, t_rate, p_type)

pop = gen_init_pop(pop_size, T, F, max_depth, method, t_rate, p_type)

print("\n===== CartPole =====")
# print("gen_program()")
# print("p1 = {}".format(p1))
# print("p2 = {}".format(p2))

print("gen_init_pop()")
for p in pop:
    print(p)
print("===== CartPole =====\n")



pendulum_info["term_growth_rate"] = .5
pendulum_info["max_depth"] = 2
pendulum_info["pop_size"] = 2

T = pendulum_info["T"]
F = pendulum_info["F"]
max_depth = pendulum_info["max_depth"]
method = pendulum_info["method"]
t_rate = pendulum_info["term_growth_rate"]
p_type = pendulum_info["program_type"]

p1 = gen_program(T, F, max_depth, method, t_rate, p_type)
p2 = gen_program(T, F, max_depth, method, t_rate, p_type)

pop = gen_init_pop(pop_size, T, F, max_depth, method, t_rate, p_type)

print("\n===== Pendulum =====")
# print("gen_program()")
# print("p1 = {}".format(p1))
# print("p2 = {}".format(p2))

print("gen_init_pop()")
for p in pop:
    print(p)
print("===== Pendulum =====\n")
