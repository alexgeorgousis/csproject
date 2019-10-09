import gym
import minerl
import matplotlib.pyplot as plt

# Experiment parameters
max_time_steps = 1000
num_episodes = 1
seed = 1015

env = gym.make("MineRLNavigateDense-v0")

for ep in range(num_episodes):

	done = False
	net_reward = 0
	action = env.action_space.noop()
	reward_measues = []
	compass_measures = []

	close_to_goal = False
	angle_threshold = 150  # indicates that the agent just passed by the goal
	init_adjust_done = False
	adjust_period = 100    # compass angle adjustment period

	env.seed(seed)
	obs = env.reset()

	for time_step in range(max_time_steps):
		_ = env.render()

		# Keep turning towards goal
		action['camera'] = [0, 0.03*obs['compassAngle']]
		
		# Check if initial adjustment is done
		if time_step > adjust_period:
			init_adjust_done = True

		# Initial adjustment is done
		if init_adjust_done:
			
			# Check if agent is close to goal
			if obs['compassAngle'] > angle_threshold:
				close_to_goal = True

			# Agent isn't close to goal
			if not close_to_goal:
				action['forward'] = 1
				action['jump'] = 1

			# Agent is close to goal
			else:
				action['forward'] = 0
				action['jump'] = 0

		# Take action, record measurements, check for termination
		obs, reward, done, info = env.step(action)
		net_reward += reward
		reward_measues.append(net_reward)
		compass_measures.append(obs['compassAngle'])

		if done:
			break

env.close()



print("Net reward: " + str((net_reward)))

# Plot measurements
plt.figure(1)
plt.plot(reward_measues, '-b')
plt.title('Reward over the episode')
plt.xlabel('time step')
plt.ylabel('reward')

plt.figure(2)
plt.plot(compass_measures, '-b')
plt.title('Compass angle over the episode')
plt.xlabel('time step')
plt.ylabel('compass angle')

plt.show()
