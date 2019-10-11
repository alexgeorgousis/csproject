import gym
import minerl
import matplotlib.pyplot as plt

# REMOVE
test = True
# /REMOVE

# Experiment parameters
max_time_steps = 3000
num_episodes = 1
seed = 1015

env = gym.make("MineRLNavigateDense-v0")

for ep in range(num_episodes):

	done = False
	net_reward = 0
	action = env.action_space.noop()
	reward_measues = []
	compass_measures = []

	# Episode stages
	adjusting = False
	init_stage = True
	running_stage = False
	close_to_goal = False
	approach_goal = False
	search_goal = False

	angle_threshold = 150   # indicates that the agent just passed by the goal
	adjust_period = 100     # compass angle adjustment period
	time_counter = 0        # a generic timer for counting duration in time steps
	cam_turn_factor = 0.03  # proportion of the compass angle, used to turn the camera towards it

	env.seed(seed)
	obs = env.reset()

	for time_step in range(max_time_steps):
		_ = env.render()

		# Wait until the compass has adjusted
		if adjusting:
			print("adjusting " + str(time_counter))
			action = env.action_space.noop()
			adjusting = time_counter < adjust_period

		# Stage 1
		if init_stage and not adjusting:
			print("init stage " + str(time_counter))
			action['camera'] = [0, cam_turn_factor*obs['compassAngle']]

			if time_counter > adjust_period:
				init_stage = False
				running_stage = True
				adjusting = True
				time_counter = 0

		# Stage 2: run until close to goal
		if running_stage and not adjusting:
			print("running_stage " + str(time_counter))

			# Run towards goal
			action['forward'] = 1
			action['jump'] = 1
			action['camera'] = [0, cam_turn_factor*obs['compassAngle']]

			# Stop when the agent surpasses the goal

			if abs(obs['compassAngle']) > angle_threshold:
				action['forward'] = 0
				action['jump'] = 0

				running_stage = False
				close_to_goal = True
				time_counter = 0

		# Stage 3: agent surpassed goal, turn towards it
		if close_to_goal and not adjusting:
			print("close to goal " + str(time_counter))
			action['camera'] = [0, cam_turn_factor*obs['compassAngle']]

			if time_counter > adjust_period:
				close_to_goal = False
				approach_goal = True
				adjusting = True
				time_counter = 0

		# Stage 4: approach goal
		if approach_goal and not adjusting:
			print("approaching goal " + str(time_counter))
			
			# Stop and move to next stage when agent surpasses goal
			if abs(obs['compassAngle']) > angle_threshold:
				action['forward'] = 0
				approach_goal = False
				search_goal = True
			else:
				action['forward'] = 1
				adjusting = True
				time_counter = 0

		# Stage 5: search for goal
		if search_goal:
			print("searching for the goal")

		# Take action, update, record measurements
		obs, reward, done, info = env.step(action)
		net_reward += reward
		time_counter += 1
		reward_measues.append(net_reward)
		compass_measures.append(obs['compassAngle'])

		# Check for termination
		if done:
			print("Done!")
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
