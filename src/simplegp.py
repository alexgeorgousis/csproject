"""
Objective: create genetic programming algorithm to solve the equation x^2 + x + 1
"""


import numpy as np
import matplotlib.pyplot as plt
import random
import copy


fset = ['+', '-', '*']
tset = ['x', '-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5' ]

n = 4         # population size
depth = 2     # initial program depth
max_gen = 500   # max number of generations to run the experiment for
solution = ['+', ['*', 'x', 'x'], ['+', 'x', '1']]  # x^2 + x + 1
fitness_goal = 0.1

def init(n):
	"""
	Creates an initial random population of the given depth,
	using the given function and terminal sets
	"""

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

	return [fitness(p, solution) for p in ps]

def select(population, fitness_scores):
	"""
	Selects a program from a population randomly, based on the fitness scores.
	"""

	probs = []     # probability of selection of each program
	selected = []  # selected program

	# Compute selection probabilities
	for i in range(len(population)):
		probs.append((random.randrange(50, 100, 1)/100) / fitness_scores[i])
		# probs.append(random.random() / fitness_scores[i])

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

	# Get node at selected point in parent 2
	p2_node = []
	get_node_at_point(p2, p2_point, p2_node)

	# Replace node in parent 1 with node in parent 2
	replace_node(p3, p1_point, p2_node[0])

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

	idx = 1
	while point > -1 and idx < len(p):
		if point == 0:
			node.append(copy.deepcopy(p[idx]))
			point -= 1
		else:
			point -= 1
			if isinstance(p[idx], list):
				point = get_node_at_point(p[idx], point, node)
		idx += 1
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

		idx += 1

	return point

def eval(exp, x):
	"""
	Evaluates an expression, using the given the value for x.
	"""

	value = 0

	# If the expression is a function
	if isinstance(exp, list):
		func = exp[0]

		# TODO: don't assume function arity
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


# Initialisation
population = init(n)
best_fitness_scores = []  # best fitness from each generation
avg_fitness_scores = []   # average fitness from each generation

# Main loop
for gen_counter in range(1, max_gen+1):

	# Compute fitness scores
	fitness_scores = batch_fitness(population, solution)
	best_fitness_scores.append(min(fitness_scores))
	avg_fitness_scores.append(sum(fitness_scores)/len(fitness_scores))

	# Display generation info
	print("\nGeneration #" + str(gen_counter))
	for idx in range(len(population)):
		print(population[idx], fitness_scores[idx])
	print()

	# Check for winner
	idx = 0
	winner_found = False
	while not winner_found and idx < len(population):
		if fitness_scores[idx] <= fitness_goal:
			winner_found = True
		else:
			idx += 1

	if winner_found:
		print("Winner!")
		print(population[idx])
		print()
		break

	# Reproduction
	p_repr = select(population, fitness_scores)


	# Mutation
	p_mutation = select(population, fitness_scores)
	mutate(p_mutation)


	# Crossover
	p1 = select(population, fitness_scores)
	p2 = select(population, fitness_scores)
	p3 = select(population, fitness_scores)
	p4 = select(population, fitness_scores)

	p1_crossover = crossover(p1, p2)
	p2_crossover = crossover(p3, p4)


	# Add programs to next generation
	population = [p_repr, p_mutation, p1_crossover, p2_crossover]
	fitness_scores = batch_fitness(population, solution)


# Plot average fitness scores
gen_counts = [i+1 for i in range(len(best_fitness_scores))]
plt.plot(gen_counts, best_fitness_scores, '-b')
plt.title('Best Fitness')
plt.ylabel('highest fitness score')
plt.xlabel('generation number')
plt.show()
