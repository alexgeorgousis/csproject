import gym
import numpy as np

# Types
Program = [str]

class Agent:
	
	def __init__(self):
		self._T = ["pa", '0', 'L', 'R']
		self._F = ["IFLTE"]
		self._pop_size = 10
		self._init_pop = self._gen_init_pop(self._pop_size)

	def _gen_init_pop(self, pop_size) -> [Program]:
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

	def _fit(p: Program) -> float:
		pass
