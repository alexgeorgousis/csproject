import gym, minerl, logging
import numpy as np
import matplotlib.pyplot as plt
from minerlgp import MineRLGP
# logging.basicConfig(level=logging.DEBUG)

# GP setup
gp = MineRLGP(pop_size=5)

# Environment setup
env = gym.make('MineRLNavigateDense-v0')

# Episode setup
obs = env.reset()
done = False
compass_angle_measures = []
net_reward = 0
net_reward_measures = [0]

# Episode main loop
while not done:
	env.render()

	action = env.action_space.noop()
	
	obs, reward, done, info = env.step(action)

	# Save measurements
	compass_angle_measures.append(obs['compassAngle'])
	net_reward += reward
	net_reward_measures.append(net_reward)

env.close()


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
