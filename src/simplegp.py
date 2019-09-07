"""
Objective: create genetic programming algorithm to solve the equation x^2 + x + 1
"""


import numpy as np
import random
import copy


fset = ['+', '-', '*']
tset = ['x', '-2', '-1', '0', '1', '2']

# # Seeds
# init_seed = 1      # initial population
# repr_seed = 2       # reproduction
# mutation_seed = 3   # mutation
# crossover_seed = 4  # crossover

n = 4      # population size
depth = 2  # program tree depth
solution = ['+', ['*', 'x', 'x'], ['+', 'x', '1']]  # x^2 + x + 1

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

	# Select program with highest probability
	max_prob = max(probs)
	for j in range(len(population)):
		if probs[j] == max_prob:
			selected = copy.deepcopy(population[j])

	return selected

def mutate(p):
	"""
	Mutates the node of a program p (tree-based mutation).
	"""

	# Randomly select a mutation point on p (node index)
	mutation_point = select_rnd_point(p)

	# Randomly mutate selected node in p
	replace_node(p, mutation_point)

def crossover(p1, p2):
	"""
	Performs crossover on two programs (parents) and returns the resulting program.
	"""

	p3 = copy.deepcopy(p1)  # crossover result

	# Select crossover points on the parents
	p1_point = select_rnd_point(p1)
	p2_point = select_rnd_point(p2)
	print(p1_point, p2_point)

	# Get node at selected point in parent 2
	p2_node = None
	get_node_at_point(p2, p2_point, p2_node)
	print("P2 node:", p2_node)

	print("\nResult\n", p3)
	return p3

def select_rnd_point(p):
	"""
	Randomly selects a node in p and returns its index.
	"""

	# Assign a probability to each node of p
	node_probs = []
	assign_node_probs(p, node_probs)

	# Select node index with highest probability
	return node_probs.index(max(node_probs))	

def get_node_at_point(p, point, node):
	"""
	Finds the node at a point in a program and loads it into a given node.
	"""

	print(p)
	print("Point =", point)
	print(node)

	idx = 1
	while point > -1 and idx < len(p):
		print("while(" + str(idx) + ")")
		if point == 0:
			node = copy.deepcopy(p[idx])
			print(node)
			point -= 1
		else:
			point -= 1
			print("point - 1 =", point)
			if isinstance(p[idx], list):
				point = get_node_at_point(p[idx], point, node)

	return point

def assign_node_probs(node, probs):
	"""
	Assigns a random probability to all the sub-nodes of a node (including terminals).
	"""

	if isinstance(node, list):
		for subnode_i in range(1, len(node)):
			probs.append(random.random())
			assign_node_probs(node[subnode_i], probs)

def replace_node(p, point, new_node=None):
	"""
	Finds and replaces the node at a point in a program, 
	either with a randomly generated node of the same depth or a given node.
	"""

	idx = 1
	while point > -1 and idx < len(p):
		if point == 0:

			# Randomly generate new node (used in mutation)
			if new_node == None:
				if isinstance(p[idx], list):
					p[idx] = gen_rnd_exp(fset, tset, 1)
				elif p[idx] in tset:
					p[idx] = gen_rnd_exp(fset, tset, 0)
			
			# Replace with given node (used in crossover)
			else:
				p[idx] = new_node

			point -= 1

		else:
			point -= 1
			if isinstance(p[idx], list):
				point = replace_node(p[idx], point, new_node)

	return point

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


"""--- Initialisation ---"""
# random.seed(init_seed)

# Generate initial population
# population = init(n)
# print("Initial population")
# for p in population: print(p)
# print()

# Compute fitness scores
# fitness_scores = batch_fitness(population, solution)
# print("Fitness scores")
# for f in fitness_scores: print(f)
# print()


"""--- Reproduction ---"""
# random.seed(repr_seed)

# Select program for reproduction
# p_repr = select(population, fitness_scores)
# print("Program for reproduction")
# print(p_repr)
# print()

# Reproduce selected program into next generation
# next_gen = [p_repr]
# print("Next generation (after reproduction)")
# for p_new in next_gen: print(p_new)
# print()


"""--- Mutation ---"""
# random.seed(mutation_seed)

# Select program for mutation
# p_mutation = select(population, fitness_scores)
# print("Program for mutation")
# print(p_mutation)
# print()

# Mutate selected program
# mutate(p_mutation)
# print("Mutated version of the program")
# print(p_mutation)
# print()

# Add mutated program into next generation
# next_gen.append(p_mutation)
# print("Next generation (after mutation)")
# for p_new in next_gen: print(p_new)
# print()


"""--- Crossover ---"""
# random.seed(crossover_seed)

# Select programs for crossover
# p1_crossover = select(population, fitness_scores)
# p2_crossover = select(population, fitness_scores)
# print("Parents for crossover\n", p1_crossover, '\n', p2_crossover, '\n')

# Perform crossover and add result to next generation
# p_crossover = crossover(p1_crossover, p2_crossover)

node = [1,2,3]
get_node_at_point(['-', ['*', 'x', '0'], ['*', '0', '-2']], 0, node)
print("Result:", node)
