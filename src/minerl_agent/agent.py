import gym
import minerl
import matplotlib.pyplot as plt

env = gym.make("MineRLNavigateDense-v0")
obs = env.reset()

done = False
net_reward = 0
reward_measues = []
compass_measures = []

for time_step in range(1000):
	_ = env.render()

	action = env.action_space.noop()

	# Keep turning towards goal
	action['camera'] = [0, 0.03*obs['compassAngle']]
	
	# Wait 100 timesteps for the compass to adjust
	if time_step > 100:

		# Stop when compass angle overshoots (passed goal)
		if abs(obs['compassAngle']) > 150:
			for i in range(time_step, 1000):
				_ = env.render()

				action['jump'] = 0
				action['forward'] = 0

				obs, reward, done, info = env.step(action)
				net_reward += reward
				reward_measues.append(reward)
				compass_measures.append(obs['compassAngle'])

				if done:
					break
			break

		# Keep running and jumping towards goal
		else:
			action['jump'] = 1
			action['forward'] = 1

	obs, reward, done, info = env.step(action)

	net_reward += reward
	reward_measues.append(reward)
	compass_measures.append(obs['compassAngle'])

	if done:
		break

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
