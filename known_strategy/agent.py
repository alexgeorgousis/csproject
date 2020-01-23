import gym
import numpy as np

# Types
Program = [str]

class Agent:
	
	def __init__(self):
		# Agent structure parameters
		self._T = ["pa", '0', 'L', 'R']
		self._F = ["IFLTE"]
		self._actions = {'left': 0, 'right': 1}

		# GP experiment parameters
		self._pop_size = 10
		self._num_eps = 100  # number of episodes to evaluate each program on

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

	def _fit(self, p: Program) -> float:
		avg_reward = 0

		net_reward = 0
		num_eps = self._num_eps

		# Make environment and run episodes
		env = gym.make("CartPole-v0")
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
		result = -1

		pa = obs[2]

		arg1 = pa if p[1] == 'pa' else 0
		arg2 = pa if p[2] == 'pa' else 0

		if arg1 <= arg2:
			arg3 = self._actions["left"] if p[3] == 'L' else self._actions["right"]
			result = arg3
		else:
			arg4 = self._actions["left"] if p[4] == 'L' else self._actions["right"]
			result = arg4

		return result
