import numpy as np
import gym, minerl

import logging
logging.basicConfig(level=logging.DEBUG)


class MineRLGP:
	"""
	A GP algorithm for the MineRL Navigate Dense environment.
	"""

	# TODO: Implement _eval(indiv)
	# 

	# GP parameters
	_pop_size = 4

	# Function and terminal sets
	_fset = []
	_tset = ['forward', 'back', 'left', 'right']

	init_pop = []  # initial population

	# Environment parameters
	_env_name = ""             # name of MineRL environment
	env = None                # MineRL environment instance
	_max_episode_steps = 200   # Maximum number of timesteps before episode termination

	def __init__(self, pop_size=4, env="MineRLNavigateDense-v0"):
		"""
		Initialise GP and MineRL environment.
		"""

		# GP parameters
		self._pop_size = pop_size
		
		# Initial GP population
		for _ in range(self._pop_size):
			self.init_pop.append([np.random.choice(self._tset)])

		# MineRL parameters
		self._env_name = env
		self.env = gym.make(self._env_name)
		self.env._max_episode_steps = self._max_episode_steps

	def fitness (self, pop):
		"""Evaluates the fitness of each individual in a population"""

		rewards = []

		for i in pop:

			# Episode setup
			obs = self.env.reset()
			done = False
			net_reward = 0

			# Episode main loop
			while not done:
				self.env.render()

				action = self._eval(i, obs)
				
				obs, reward, done, info = self.env.step(action)

				net_reward += reward

			# Store individual net reward
			rewards.append(net_reward)

		return rewards

	def _eval(self, indiv, obs):
		"""Evaluates an individual to extract an action, given an observation."""

		action = self.env.action_space.noop()
		action[indiv[0]] = 1

		return action



# TESTING #
gp = MineRLGP(pop_size=4)
fitness_scores = gp.fitness(gp.init_pop)
gp.env.close()

print('\n')
print(gp.init_pop)
print(fitness_scores)
print()
