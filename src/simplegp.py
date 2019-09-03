"""
Objective: create genetic programming algorithm to solve the equation x^2 + x + 1
"""

# TODO: 

import numpy as np
import random


fset = ['+', '-', '*']
tset = ['x', '-2', '-1', '0', '1', '2']

seed = 2   # random seed (to get consistent initial population)
n = 4      # population size
depth = 2  # program tree depth
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

	# Check for winner (error < winning_condition)
	for i in range(len(scores)):
		if scores[i] < 0.1:
			print("We've got a winner")
			print(ps[i])

	return scores

def select(population, fitness_scores):
	"""
	Selects a program from a population randomly, based on the fitness scores.
	"""

	probs = []     # probability of selection of each program
	selected = []  # selected program

	# Compute selection probabilities
	for i in range(len(population)):
		probs.append(random.random() / fitness_scores[i])

	# Select program with higest probability
	max_prob = max(probs)
	for j in range(len(population)):
		if probs[j] == max_prob:
			selected = population[j]

	return selected

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

# Generate initial population
population = init(n)
print("Initial population")
print(population)
print()

# Compute fitness scores
fitness_scores = batch_fitness(population, solution)
print("Fitness scores")
print(fitness_scores)
print()

# Select program for reproduction
p_repr = select(population, fitness_scores)
print("Program for reproduction")
print(p_repr)
print()

# Reproduce selected program into next generation
next_gen = [p_repr]
print("Next generation (after reproduction)")
print(next_gen)
print()

# Select program for mutation
p_mutation = select(population, fitness_scores)
print("Program for mutation")
print(p_mutation)
print()

# Mutate selected program into next generation
