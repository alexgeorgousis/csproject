import gym
import numpy as np

# Types
Program = [str]
Population = [Program]

class Agent:
	
	def __init__(self):
		# Agent structure parameters
		self._T_values = ["pa", 'pv', '0.0', '0.025']
		self._T_actions = ['L', 'R']
		self._F = ["IFLTE"]
		self._program_depth = 2
		self._actions = {'L': 0, 'R': 1}

		# GP experiment parameters
		self._pop_size = 100
		self._num_eps = 100  # number of episodes to evaluate each program on
		self._max_gens = 10   # max number of generations to evolve
		self._term_score = 195.0  # fitness score termination criterion

		self._init_pop = self._gen_init_pop()
		self._best_program = []

	def train(self):
		best_program = []

		# Evolve generations
		current_pop = self._init_pop
		for gen_idx in range(self._max_gens):
			print("\nGeneration {}...".format(gen_idx+1))

			scores = self._batch_fit(current_pop)

			# Check termination criteria before evolving next generation
			max_score = max(scores)
			if max_score >= self._term_score:
				best_program = current_pop[scores.index(max_score)]
				break

			# Selection & reproduction
			next_pop = [self._select(current_pop, scores) for _ in range(self._pop_size)]
			current_pop = next_pop

		# If a solution wasn't found before reaching the last generation
		# pick the best program from the last generation as the solution.
		if gen_idx >= self._max_gens-1:
			last_scores = self._batch_fit(current_pop)
			max_score_idx = last_scores.index(max(last_scores))
			best_program = current_pop[max_score_idx]

		self._best_program = best_program

	def run(self):
		if self._best_program == []:
			self.train()

		print("\nBest program after training:")
		print(self._best_program)

		env = gym.make("CartPole-v0")

		net_reward = 0

		for _ in range(self._num_eps):
			ep_reward = 0
			done = False
			obs = env.reset()

			while not done:
				env.render()
				action = self._eval(self._best_program, obs)
				obs, reward, done, _ = env.step(action)
				ep_reward += reward
			net_reward += ep_reward

		print("\nAverage reward over {} trials: {}".format(self._num_eps, net_reward/self._num_eps))
		env.close()

	def _gen_init_pop(self) -> Population:
		n = self._pop_size
		pop = [self._gen_program(self._program_depth) for _ in range(n)]
		return pop

	def _gen_program(self, d: int) -> Program:
		"""
		Generates a program of arbitrary depth d.
		"""

		p = []
		
		func = np.random.choice(self._F)
		arg1 = np.random.choice(self._T_values)
		arg2 = np.random.choice(self._T_values)

		if d <= 1:
			arg3 = np.random.choice(self._T_actions)
			arg4 = np.random.choice(self._T_actions)
		else:
			arg3 = self._gen_program(d-1)
			arg4 = self._gen_program(d-1)

		p = [func, arg1, arg2, arg3, arg4]
		return p

	def _batch_fit(self, pop: Population) -> [float]:
		"""
		Computes the fitness of a population of programs.

		- pop: population (list of programs)
		- return: list of fitness scores
		"""

		fit_scores = []

		env = gym.make("CartPole-v0")
		fit_scores = [self._fit(p, env) for p in pop]
		env.close()

		return fit_scores

	def _fit(self, p: Program, env) -> float:
		"""
		Computes the average fitness of a program over 
		a certain number of runs of the environment.

		- p: program
		- env: gym environment object
		- return: fitness score
		"""

		avg_reward = 0

		net_reward = 0
		num_eps = self._num_eps

		# Run episodes
		for _ in range(num_eps):
			ep_reward = 0
			done = False
			obs = env.reset()

			# Run single episode
			while not done:
				action = self._eval(p, obs)
				obs, rew, done, _ = env.step(action)
				ep_reward += rew
				
			net_reward += ep_reward

		avg_reward = net_reward / num_eps
		return avg_reward

	def _eval(self, p:Program, obs:[float]) -> int:
		"""
		Interpreter: this function evaluates a program and outputs
		the action it takes, parameterised by an observation from the environment.

		- p: program to evaluate
		- obs: gym environment observation object
		- return: action (0 or 1 for CartPole-v0)
		"""
		
		action = -1

		pa = obs[2]
		pv = obs[3]

		# Evaluate arguments 1 and 2
		if p[1] == 'pa':
			arg1 = pa
		elif p[1] == 'pv':
			arg1 = pv
		else:
			arg1 = float(p[1])

		if p[2] == 'pa':
			arg2 = pa
		elif p[2] == 'pv':
			arg2 = pv
		else:
			arg2 = float(p[2])

		# Evaluate arguments 3 and 4
		arg3 = self._eval(p[3], obs) if type(p[3]) is list else self._actions[p[3]]
		arg4 = self._eval(p[4], obs) if type(p[4]) is list else self._actions[p[4]]

		# Evaluate IFLTE(arg1, arg2, arg3, arg4)
		if arg1 <= arg2:
			action = arg3
		else:
			action = arg4

		return action


	# Genetic operators #
	def _select(self, pop: Population, fit_scores: [float]) -> Program:
		"""
		Fitness Proportionate Selection (Roulette Wheel Selection)

		pop: population
		f_scores: fitness scores
		"""

		selected = []

		F = sum(fit_scores)
		r = np.random.uniform(0, F)

		# Simulate roulette wheel with r as the fixed point
		counter = 0
		for i in range(len(fit_scores)):
			counter += fit_scores[i]
			if counter > r:
				selected = pop[i]
				break

		return selected
