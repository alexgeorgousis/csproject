"""
Objective: generate 1 valid program of depth 1 randomly using the Full method.
"""

# TODO: implement fitness function

import random


fset = ['+', '-', '*']
tset = ['x', '-2', '-1', '0', '1', '2']

n = 4
depth = 2

def init(n):
	population = []

	for i in range(n):
		population.append(gen_rnd_exp(fset, tset, depth))
	
	return population

def fitness(p):
	"""
	Computes the fitness of the program p.
	"""
	

def gen_rnd_exp(fset, tset, max_depth):
	exp = ""

	if max_depth == 0:
		
		return choose_rnd_element(tset)

	else:
		func = choose_rnd_element(fset)
		arg1 = gen_rnd_exp(fset, tset, max_depth-1)
		arg2 = gen_rnd_exp(fset, tset, max_depth-1)

		exp = func + '(' + arg1 + ',' + arg2 + ')'

	return exp

def choose_rnd_element(set):
	index = random.randint(0, len(set)-1)
	return set[index]


population = init(n)
print(population)
print(fitness(population[0]))
