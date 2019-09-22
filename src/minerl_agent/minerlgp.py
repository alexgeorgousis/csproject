import numpy as np
import gym, minerl

# import logging
# logging.basicConfig(level=logging.DEBUG)


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
	_tset = ['', 'forward']

	_init_pop = []  # initial population

	# Environment parameters
	_env_name = ""             # name of MineRL environment
	_env = None                # MineRL environment instance
	_max_episode_steps = 2000  # Maximum number of timesteps before episode termination

	def __init__(self, pop_size=4, env="MineRLNavigateDense-v0"):
		"""
		Initialise GP and MineRL environment.
		"""

		# GP parameters
		self._pop_size = pop_size
		
		# Initial GP population
		for _ in range(self._pop_size):
			self._init_pop.append([np.random.choice(self._tset)])

		# MineRL parameters
		self._env_name = env
		self._env = gym.make(self._env_name)
		self._env._max_episode_steps = self._max_episode_steps

	def fitness (self, indiv):
		"""Evaluates the fitness of an individual"""

		# Episode setup
		obs = self._env.reset()
		done = False
		net_reward = 0

		# Episode main loop
		while not done:
			self._env.render()

			action = self._eval(indiv)
			
			obs, reward, done, info = self._env.step(action)

			# Save measurements
			net_reward += reward

		print("Net Reward: " + str(net_reward))

	def _eval(indiv):
		"""Evaluates an individual to extract an action."""




# TESTING #
gp = MineRLGP(pop_size=1)
gp.fitness(gp._init_pop[0])
gp._env.close()
