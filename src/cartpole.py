import gym

env = gym.make('CartPole-v0')
obs = env.reset()

while True:
	env.render()
	
	# Perform action
	action = env.action_space.sample()
	obs, reward, done, info = env.step(action)

	if done:
		env.reset()

env.close()
