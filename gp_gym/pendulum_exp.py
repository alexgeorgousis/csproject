"""
This module is dedicated to running controlled experiments on the Pendulum-v0
environment.
"""

import gym
import matplotlib.pyplot as plt
import numpy as np



""" # ----- Experiment 1: random agent -----
    
    def random_agent(num_trials, num_eps):
        avg_scores = []

        env = gym.make("Pendulum-v0")
        for i in range(num_trials):
            trial_reward = 0

            # Run episodes
            for _ in range(num_eps):
                ep_reward = 0
                done = False
                obs = env.reset()

                # Run single episode
                while not done:
                    action = env.action_space.sample()
                    obs, rew, done, _ = env.step(action)
                    
                    # In pendulum, the reward is returned in an array.
                    # So, cast it to a float.
                    rew = float(rew)
                    ep_reward += rew
                    
                trial_reward += ep_reward
            
            avg_scores.append(trial_reward/num_eps)

        return avg_scores
    
    num_trials = 50
    num_eps = 10
    scores = random_agent(num_trials, num_eps)
    # print(sum(scores)/len(scores))

    plt.plot(range(1, len(scores)+1), scores)
    # plt.title("Random agent", fontdict={"size": 16})
    plt.ylabel("Average Cost", fontdict={"size": 12})
    plt.xlabel("Trial", fontdict={"size": 12})
    # plt.show()
    # ----- Experiment 1: random agent -----
"""



""" # ----- Experiment 2: single-constant agent -----

    def constant_agent(num_trials, num_eps, actions):
        avg_scores = []

        env = gym.make("Pendulum-v0")
        for i in range(num_trials):
            trial_reward = 0

            # Run episodes
            for _ in range(num_eps):
                ep_reward = 0
                done = False
                obs = env.reset()

                # Run single episode
                while not done:
                    obs, rew, done, _ = env.step([np.random.choice(actions)])
                    
                    # In pendulum, the reward is returned in an array.
                    # So, cast it to a float.
                    rew = float(rew)
                    ep_reward += rew
                    
                trial_reward += ep_reward

            avg_scores.append(trial_reward/num_eps)

        return avg_scores

    # Experiment 2.1: determine best actions
    # actions = np.arange(-2.0, 2.1, 0.1)

    # num_trials = 1
    # num_eps = 10

    # # Get best action
    # for i in range(10):
    #     scores = [constant_agent(num_trials, num_eps, [a]) for a in actions]
    #     best_action = actions[scores.index(max(scores))]

    # plt.plot(actions, scores)
    # plt.title("Constant agent", fontdict={"size": 16})
    # plt.ylabel("Cost", fontdict={"size": 12})
    # plt.xlabel("Action", fontdict={"size": 12})
    # plt.show()

    # Experiment 2.2: test constant agent
    # num_trials = 50
    # num_eps = 10

    actions = np.arange(-0.10, 0.10, 0.01)
    avg_costs = constant_agent(num_trials, num_eps, actions)
    # total_avg_cost = sum(avg_costs) / num_trials
    # print("Average cost: {}".format(total_avg_cost))

    plt.plot(range(1, num_trials+1), avg_costs)
    # plt.ylabel("Average Cost", fontdict={"size": 12})
    # plt.xlabel("Trial", fontdict={"size": 12})
    plt.legend(["Random agent", "Constant agent"])
    plt.show()
    # ----- Experiment 2: single-constant agent -----
"""



""" # ----- Experiment 3: single obs state agent -----
    The agent chooses a single obs value as its action.


    def exp3(num_trials, num_eps, obs_idx):
        avg_trial_scores = []

        env = gym.make("Pendulum-v0")
        for _ in range(num_trials):
            trial_reward = 0

            # Run episodes
            for _ in range(num_eps):
                ep_reward = 0
                done = False
                obs = env.reset()

                # Run single episode
                while not done:
                    obs, rew, done, _ = env.step([obs[obs_idx] / np.random.choice(np.arange(0.01, 0.20, 0.01))])
                    
                    # In pendulum, the reward is returned in an array.
                    # So, cast it to a float.
                    rew = float(rew)
                    ep_reward += rew
                    
                trial_reward += ep_reward

            avg_trial_scores.append(trial_reward/num_eps)

        return avg_trial_scores

    num_trials = 50
    num_eps = 10

    # costheta
    avg_costs_cos = exp3(num_trials, num_eps, 0)
    print("Total average cost (costheta): {}".format(sum(avg_costs_cos)/num_trials))

    # # sintheta
    # avg_costs_sin = exp3(num_trials, num_eps, 1)
    # print("Total average cost (sintheta): {}".format(sum(avg_costs_sin)/num_trials))

    # # thetadot
    # avg_costs_dot = exp3(num_trials, num_eps, 2)
    # print("Total average cost (thetadot): {}".format(sum(avg_costs_dot)/num_trials))

    plt.plot(range(1, num_trials+1), avg_costs_cos)
    # plt.plot(range(1, num_trials+1), avg_costs_sin)
    # plt.plot(range(1, num_trials+1), avg_costs_dot)
    plt.legend(["cos", "sin", "dot"])
    plt.xlabel("trial", fontdict={"size": 12})
    plt.ylabel("cost", fontdict={"size": 12})
    plt.show()
"""
