import numpy as np
import gym, minerl


class MineRLGP:
	"""
	A GP algorithm for the MineRL Navigate Dense environment.
	"""

	# TODO:
	# Finish implementing __init__ to create the initial population. 

	# Parameters
	_pop_size = 4

	# Function and terminal sets
	_fset = []
	_tset = ['', 'forward']

	_init_pop = []  # initial population
	_env_name = ""  # name of MineRL environment
	_env = None     # MineRL environment instance

	def __init__(self, pop_size=4, env="MineRLNavigateDense-v0"):
		"""
		Sets up the GP parameters.
		Creates an initial population.
		Loads the MineRL environment.
		"""

		self._pop_size = pop_size
		self._env_name = env
		
		for _ in range(self._pop_size):
			self._init_pop.append([np.random.choice(self._tset)])

		self._env = gym.make(self._env_name)

	def fitness (self, indiv):
		"""Evaluates the fitness of an individual"""



# TESTING #
gp = MineRLGP(pop_size=1)
