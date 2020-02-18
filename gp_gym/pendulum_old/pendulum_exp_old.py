"""
This module is dedicated to running controlled experiments on the Pendulum-v0
environment.
"""

import gym
import matplotlib.pyplot as plt
import numpy as np


"""
# ----- Experiment 1: random agent -----

    # Experiment parameters
    num_eps = 200

    # Run experiment
    env = gym.make("Pendulum-v0")
    net_reward = 0.0

    for _ in range(num_eps):
        obs = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            obs, reward, done, _ = env.step([obs[1]])
            ep_reward += reward
        net_reward += ep_reward
    env.close()

    # Print result
    print("Average reward over {} episodes: {}".format(num_eps, net_reward/num_eps))

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


"""# ----- Experiment 4: GP agent -----

    from pendulum_info_old import info
    from pendulum import_old Pendulum


    # GP parameters
    info["max_depth"] = 1
    info["term_growth_rate"] = 0.5
    info["num_eps"] = 2
    info["num_time_steps"] = 500
    info["pop_size"] = 100
    info["max_gens"] = 10
    info["term_fit"] = -500
    info["mutation_rate"] = 0.1

    # Train agent
    agent = Pendulum(info)
    p, avg_gen_fit = agent.train()
    print("\nProgram:\n{}".format(p))

    # Run experiment
    env = gym.make("Pendulum-v0")
    net_reward = 0.0

    for i in range(100):
        # env.render()
        obs = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            obs, reward, done, _ = env.step([agent.eval(p, obs)])
            ep_reward += reward
        net_reward += ep_reward
    env.close()

    # Print result
    print("Average reward: {}".format(net_reward/100))
"""

"""# ----- Experiment 5: mutation -----


    from pendulum_info import info
    from pendulum import Pendulum


    # GP parameters
    info["max_depth"] = 1
    info["term_growth_rate"] = 0.5
    info["num_eps"] = 2
    info["num_time_steps"] = 500
    info["pop_size"] = 100
    info["max_gens"] = 10
    info["term_fit"] = -500

    avg_changes = []
    for i in range(10):
        
        print("Without mutation #{}".format(i+1))

        # Train agent
        info["mutation_rate"] = 0.0
        agent = Pendulum(info)
        p, avg_gen_fit = agent.train()

        # Compute avg show the rate of change of fitness from generation to generation
        avg_fit_change = np.mean(np.abs(np.diff(avg_gen_fit)))
        # print("Average fitness change: {}".format(avg_fit_change))
        avg_changes.append(avg_fit_change)

    print("---------------------------------------------")

    avg_changes_mut = []
    for j in range(10):
        
        print("\nWith mutation #{}".format(j+1))

        # Train agent
        info["mutation_rate"] = 0.1
        agent = Pendulum(info)
        p, avg_gen_fit = agent.train()

        # Compute avg show the rate of change of fitness from generation to generation
        avg_fit_change = np.mean(np.abs(np.diff(avg_gen_fit)))
        # print("Average fitness change: {}".format(avg_fit_change))
        avg_changes_mut.append(avg_fit_change)

    print(avg_changes)
    print(avg_changes_mut)
    print("\nAverage change without mutation: {}".format(np.mean(avg_changes)))
    print("Average change with mutation: {}".format(np.mean(avg_changes_mut)))
"""
