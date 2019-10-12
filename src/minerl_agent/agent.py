import gym
import minerl
import matplotlib.pyplot as plt

# Experiment parameters
max_time_steps = 6000
num_episodes = 1
seed = 1016

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

	# Search grid representation
	search_actions = []
	search_counter = 0

	for i in range(3, 300, 2):
		search_actions.append({'forward': 1})
		search_actions.append({'camera': [0, -90]})

		for _ in range(i-2):
			search_actions.append({'forward': 1})
		search_actions.append({'camera': [0, -90]})

		for _ in range(i-1):
			search_actions.append({'forward': 1})
		search_actions.append({'camera': [0, -90]})

		for _ in range(i-1):
			search_actions.append({'forward': 1})
		search_actions.append({'camera': [0, -90]})

		for _ in range(i-1):
			search_actions.append({'forward': 1})

	print_msg = True        # used to stop printing console messages
	angle_threshold = 150   # indicates that the agent just passed by the goal
	adjust_period = 100     # compass angle adjustment period
	time_counter = 0        # a generic timer for counting duration in time steps
	cam_turn_factor = 0.03  # proportion of the compass angle, used to turn the camera towards it

	env.seed(seed)
	obs = env.reset()

	for time_step in range(max_time_steps):
		print(time_step)

		_ = env.render()

		# Wait until the compass has adjusted
		if adjusting:
			if print_msg:
				print("Adjusting...")
				print_msg = False			

			action = env.action_space.noop()
			adjusting = time_counter < adjust_period

		# Stage 1
		if init_stage and not adjusting:
			
			if print_msg:
				print("\nStage 1: turn towards goal")
				print_msg = False

			action['camera'] = [0, cam_turn_factor*obs['compassAngle']]

			if time_counter > adjust_period:
				init_stage = False
				running_stage = True
				time_counter = 0
				print_msg = True

		# Stage 2: run until close to goal
		if running_stage and not adjusting:
			
			if print_msg:
				print("\nStage 2: run towards goal")
				print_msg = False

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
				print_msg = True

		# Stage 3: agent surpassed goal, turn towards it
		if close_to_goal and not adjusting:
			
			if print_msg:
				print("\nStage 3: turn towards goal")
				print_msg = False

			action['camera'] = [0, cam_turn_factor*obs['compassAngle']]

			if time_counter > adjust_period:
				close_to_goal = False
				approach_goal = True
				adjusting = True
				time_counter = 0
				print_msg = True

		# Stage 4: approach goal
		if approach_goal and not adjusting:

			if print_msg:
				print("\nStage 4: reach goal")
				print_msg = False
			
			# Stop and move to next stage when agent surpasses goal
			if abs(obs['compassAngle']) > angle_threshold:
				action['forward'] = 0
				approach_goal = False
				search_goal = True
				time_counter = 0
				print_msg = True
			else:
				action['forward'] = 1
				adjusting = True
				time_counter = 0
				print_msg = True

		# Stage 5: search for goal
		if search_goal:
			if print_msg:
				print("Stage 5: searching...")
				print_msg = False

			adjust_period = 5
			action = env.action_space.noop()

			if time_counter > adjust_period:
				action = search_actions[search_counter]
				action['jump'] = 1
				time_counter = 0

				if search_counter + 1 < len(search_actions):
					search_counter += 1
				else:
					action = env.action_space.noop()


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
