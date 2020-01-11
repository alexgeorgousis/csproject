"""
Manually tests each strategy.
"""

import gym


num_trials = 50000  # number of trials to run the experiment for
L = 0
R = 1
strategies = [
	{'a': "pa", 'b': 0, "then_b": L, "else_b": R},
	{'a': 0, 'b': 'pa', "then_b": R, "else_b": L},
	{'a': "pa", 'b': 0, "then_b": R, "else_b": L},
	{'a': 0, 'b': "pa", "then_b": L, "else_b": R},
	{'a': "pa", 'b': 0, "then_b": L, "else_b": L},
	{'a': "pa", 'b': 0, "then_b": R, "else_b": R},
	{'a': 0, 'b': "pa", "then_b": L, "else_b": L},
	{'a': 0, 'b': "pa", "then_b": R, "else_b": R},
]

env = gym.make('CartPole-v0')
obs = env.reset()

avg_rewards = []  # mean reward of each strategy
for i in range(len(strategies)):
	print("Running strategy {}...".format(i+1))

	net_reward = 0
	for _ in range(num_trials):
		done = False
		trial_reward = 0

		while not done:
			# env.render()

			action = 0
			pa = obs[2]

			# Interpret strategy
			s = strategies[i]
			check = False

			if s['a'] == "pa":
				check = pa <= s['b']
			else:
				check = s['a'] <= pa

			if check:
				action = s['then_b']
			else:
				action = s['else_b']
			
			obs, reward, done, _ = env.step(action)
			trial_reward += reward

			if done:
				obs = env.reset()
		
		net_reward += trial_reward

	avg_rewards.append(net_reward/num_trials)

print("\nAverage reward over {} trials: {}".format(num_trials, avg_rewards))
env.close()
