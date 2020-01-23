import gym
import numpy as np

# Types
Program = [str]
Population = [Program]

class Agent:
	
	def __init__(self):
		# Agent structure parameters
		self._T = ["pa", '0.05', 'L', 'R']
		self._F = ["IFLTE"]
		self._actions = {'left': 0, 'right': 1}

		# GP experiment parameters
		self._pop_size = 10
		self._num_eps = 100  # number of episodes to evaluate each program on
		self._max_gens = 1   # max number of generations to evolve

		self._init_pop = self._gen_init_pop(self._pop_size)
		self._best_program = ['IFLTE', 'pa', '-0.05', 'L', 'R']

	def train(self):
		current_pop = self._init_pop

		# Evolve generations
		for gen_idx in range(self._max_gens):
			print("Generation {}".format(gen_idx+1))

			scores = self._batch_fit(current_pop)

			# Selection & reproduction
			next_pop = [self._select(current_pop, scores) for _ in range(self._pop_size)]
			current_pop = next_pop

		last_scores = self._batch_fit(current_pop)
		max_score_idx = last_scores.index(max(last_scores))
		self._best_program = current_pop[max_score_idx]

	def run(self):
		if self._best_program == []:
			self.train()

		print(self._best_program)
		env = gym.make("CartPole-v0")

		net_reward = 0
		done = False
		obs = env.reset()

		while True:
			env.render()
			action = self._eval(self._best_program, obs)
			obs, reward, done, _ = env.step(action)
			net_reward += reward

		print(net_reward)
		env.close()

	def _gen_init_pop(self, pop_size) -> Population:
		pop = []

		for i in range(pop_size):
			p = []

			func = np.random.choice(self._F)
			arg1 = np.random.choice(self._T[:2])
			arg2 = self._T[0] if arg1 == self._T[1] else self._T[1]
			arg3 = np.random.choice(self._T[2:])
			arg4 = self._T[2] if arg3 == self._T[3] else self._T[3]

			p = [func, arg1, arg2, arg3, arg4]

			pop.append(p)
		return pop

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
		result = -1

		pa = obs[2]

		arg1 = pa if p[1] == 'pa' else float(p[1])
		arg2 = pa if p[2] == 'pa' else float(p[2])

		if arg1 <= arg2:
			arg3 = self._actions["left"] if p[3] == 'L' else self._actions["right"]
			result = arg3
		else:
			arg4 = self._actions["left"] if p[4] == 'L' else self._actions["right"]
			result = arg4

		return result


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
