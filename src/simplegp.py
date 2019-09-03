"""
Objective: create genetic programming algorithm to solve the equation x^2 + x + 1
"""

# TODO: 

import numpy as np
import random


fset = ['+', '-', '*']
tset = ['x', '-2', '-1', '0', '1', '2']

seed = 2
n = 4
depth = 2
solution = ['+', ['*', 'x', 'x'], ['+', 'x', 1]]  # x^2 + x + 1

def init(n):
	population = []

	for i in range(n):
		population.append(gen_rnd_exp(fset, tset, depth))
	
	return population

def fitness(p, solution):
	"""
	Computes the fitness of the program p against the solution.
	"""

	errors = []

	for x in np.arange(-1, 1.1, 0.1):
		p_out = eval(p, x)
		s_out = eval(solution, x)
		errors.append(abs(s_out - p_out))

	return sum(errors)

def batch_fitness(ps, solution):
	"""
	Computes the fitness of a list of programs, ps, against the solution.
	"""

	scores = [fitness(p, solution) for p in ps]
	return scores

def eval(exp, x):
	"""
	Evaluates an expression, using the given the value for x.
	"""

	value = 0

	# If the expression is a function
	if isinstance(exp, list):
		func = exp[0]

		# TODO: don't assume function arity (num of arguments)
		arg1 = eval(exp[1], x)
		arg2 = eval(exp[2], x)

		value = apply(func, arg1, arg2)
	
	# If the expression is a terminal
	else:
		if exp == 'x':                   # variable
			value = x
		elif isinstance(int(exp), int):  # constant
			value = int(exp)

	return value
	
# TODO: don't assume function arity
def apply(func, arg1, arg2):
	"""
	Applies function func to arguments.
	"""

	result = 0

	if func in fset:
		if func == '+':
			result = arg1 + arg2
		elif func == '-':
			result = arg1 - arg2
		elif func == '*':
			result = arg1 * arg2

	return result

def gen_rnd_exp(fset, tset, max_depth):
	exp = ""

	if max_depth == 0:
		
		return choose_rnd_element(tset)

	else:
		func = choose_rnd_element(fset)
		arg1 = gen_rnd_exp(fset, tset, max_depth-1)
		arg2 = gen_rnd_exp(fset, tset, max_depth-1)

		exp = [func, arg1, arg2]

	return exp

def choose_rnd_element(set):
	index = random.randint(0, len(set)-1)
	return set[index]


random.seed(seed)
population = init(n)
print("Fitness scores: " + str(batch_fitness(population, solution)))
