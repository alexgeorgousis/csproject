import gym
import numpy as np
from numpy import random

class Agent:
	def __init__(self):

		self.n = 50                   # population size
		self.m = 5                    # individual length (number of actions)
		self.max_gens = 50            # maximum num of gens to evolve
		self.trials = 100              # number of trials to evaluate each program on
		self.gens = []
		self.final_agent = ""         # best individual of last generation
		self.mutation_ratio = 80      # amount of individuals to mutate
		self.crossover_ratio = 20     # amount of individuals to crossover

		# Generate initial population
		init_pop = [[random.randint(2, size=self.m), 0.0] for i in range(self.n)]
		self.gens.append(init_pop)


	def train(self):
		
		for gen_idx in range(self.max_gens):
			pop = self.gens[gen_idx]

			# Assign fitness scores to individuals
			pop_scores = self.pop_fit(pop)
			for idx in range(len(pop_scores)):
				pop[idx][1] = pop_scores[idx]

			# Evolve next generation
			if gen_idx < self.max_gens-1:
				self.gens.append(self.evolve(pop))

		final_gen = self.gens[-1]
		self.final_agent = self.select(final_gen)


	def pop_fit(self, pop):
		pop_rewards = []
		env = gym.make("CartPole-v0")

		for indiv_idx in range(self.n):
			obs = env.reset()
			net_reward = 0

			# Individual evaluation
			for i in range(self.trials):
				done = False
				action_idx = 0
				trial_reward = 0

				while not done:
					action = pop[indiv_idx][0][action_idx]
					obs, reward, done, _ = env.step(action)
					trial_reward += reward
					action_idx = (action_idx + 1) % self.m
					if done:
						obs = env.reset()
				
				net_reward += trial_reward

			# Record average reward
			pop_rewards.append(net_reward / self.trials)
		
		env.close()
		return pop_rewards


	def evolve(self, pop):
		next_gen = []

		# Mutation
		for j in range(self.mutation_ratio):
			indiv = self.select(pop)
			indiv = self.mutate(indiv)
			next_gen.append([indiv, 0.0])

		# Crossover
		for k in range(self.crossover_ratio):
			parent1 = self.select(pop)
			parent2 = self.select(pop)
			child = self.crossover(parent1, parent2)
			next_gen.append([child, 0.0])

		return next_gen


	def select(self, pop):
		max_score_idx = 0
		for i in range(1, len(pop)):
			if pop[i][1] > pop[max_score_idx][1]:
				max_score_idx = i

		return pop[max_score_idx][0]


	def mutate(self, indiv):
		mutated = indiv.copy()

		mutation_point = random.randint(len(mutated))
		if mutated[mutation_point] == 0:
			mutated[mutation_point] = 1
		else:
			mutated[mutation_point] = 0

		return mutated


	def crossover(self, parent1, parent2):
		return np.concatenate((parent1[:self.m//2], parent2[self.m//2:]))


	def run(self):
		env = gym.make('CartPole-v0')
		obs = env.reset()
		net_reward = 0
		num_trials = 50000

		for i in range(num_trials):
			done = False
			action_idx = 0
			trial_reward = 0

			while not done:
				# env.render()
				action = self.final_agent[action_idx]
				obs, reward, done, _ = env.step(action)
				trial_reward += reward
				action_idx = (action_idx + 1) % self.m
				if done:
					obs = env.reset()
			
			net_reward += trial_reward

		print("\nAverage reward over {} trials: {}".format(num_trials, str(net_reward/num_trials)))
		env.close()
		