import gym, minerl, logging
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.DEBUG)

env = gym.make('MineRLNavigateDense-v0')

obs = env.reset()
done = False

compass_angle_measures = []
net_reward = 0
net_reward_measures = [0]

while not done:
	env.render()

	action = env.action_space.noop()
	action['camera'] = [0, 0.03 * obs['compassAngle']]
	action['forward'] = 1
	action['jump'] = 1
	action['attack'] = 1
	
	obs, reward, done, info = env.step(action)

	# Save measurements
	compass_angle_measures.append(obs['compassAngle'])
	net_reward += reward
	net_reward_measures.append(net_reward)


# Plot compass angle over time
plt.figure(1)
plt.plot(list(enumerate(compass_angle_measures)), compass_angle_measures, '-b')
plt.title('Compass Angle over the course of the episode')
plt.ylabel('compass angle')
plt.xlabel('time step')

# Plot reward over time
plt.figure(2)
reward_chunks = net_reward_measures[::10]
plt.plot(list(enumerate(reward_chunks)), reward_chunks, '-r')
plt.title('Net reward over the course of the episode')
plt.ylabel('net reward')
plt.xlabel('time step')

plt.show()
