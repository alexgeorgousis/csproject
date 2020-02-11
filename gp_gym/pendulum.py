import gym
import numpy as np
from gp_gym import gen_init_pop, select, IFLTE
import time


class Pendulum:
    """
    This class implements a GP agent for the Pendulum-v0 gym environment.
    """

    def __init__(self, info):
        self.env_name = info["env_name"]

        # Program structure
        self.p_type = info["program_type"]
        self.T = info["T"]
        self.F = info["F"]
        self.max_depth = info["max_depth"]
        self.t_rate = info["term_growth_rate"]
        self.method = info["method"]

        # GP parameters
        self.pop_size = info["pop_size"]
        self.num_eps = info["num_eps"]                # number of episodes to evaluate each program on
        self.num_time_steps = info["num_time_steps"]  # number of time steps per episode
        self.max_gens = info["max_gens"]
        self.term_fit = info["term_fit"]


    def train(self):
        best_program = None
        gen_scores = []  # used to record avg fitness of each generation

        # Generate initial population
        current_pop = gen_init_pop(self.pop_size, self.T, self.F, self.max_depth, self.method, self.t_rate, self.p_type)

        # Evolution loop
        gen_idx = 0
        while (not best_program) and (gen_idx < self.max_gens):

            # Evaluate population fitness
            start_time = time.time()
            fit_scores = self.batch_fit(current_pop, self.num_time_steps, self.num_eps)
            end_time = time.time()

            # Store average population fitness
            print("Gen {}: {}".format(gen_idx+1, np.mean(fit_scores)))
            gen_scores.append(np.mean(fit_scores))
            print("Time: {}\n".format(end_time - start_time))

            # Check termination criteria
            max_fitness = max(fit_scores)
            if (max_fitness >= self.term_fit) or (gen_idx >= self.max_gens - 1):
                best_program = current_pop[fit_scores.index(max_fitness)]

            # Evolve next generation
            else:
                current_pop = [select(current_pop, fit_scores) for _ in range(self.pop_size)]
                gen_idx += 1

        return best_program, gen_scores


    def batch_fit(self, pop, num_time_steps, num_eps):
        """
        Computes the fitness score of every program in a population.

        pop: population of programs
        num_eps: number of episodes to evaluate each program on
        """

        env = gym.make(self.env_name)
        scores = [self.fit(p, num_time_steps, num_eps, env=env) for p in pop]
        env.close()
        return scores


    def fit(self, p, num_time_steps, num_eps, env=None, render=False):
        """
        Computes the average reward of a program.

        env: gym environment object
        p: program to evaluate
        num_time_steps: number of time steps to run each episode for
        num_eps: number of episodes to run program for
        return: fitness score (float)
        """

        avg_score = 0.0
        ep_score = 0.0

        if not env:
            env = gym.make(self.env_name)

        for i in range(num_eps):

            # Run single episode
            obs = env.reset()
            for j in range(num_time_steps):

                if render:
                    env.render()

                action = self.eval(p, obs)
                obs, reward, _, _ = env.step([action])
                ep_score += reward
            # End run single episode

        avg_score = ep_score / num_eps
        
        return avg_score


    def eval(self, p, obs):
        """
        Interprets a program and evaluates it to a floating point number that 
        represents an action, given an observation from the environment.

        Note: in Pendulum-v0 actions are arrays, but this function returns a float. 
        So, it needs to be placed in an array before being sent to the environment.

        p: program to interpret
        obs: observation : [float]
        return: float
        """

        result = 0.0

        # Terminals
        if type(p) is not list:
            terminal = self.T[p]

            # Observation variables
            if terminal["token"] == "ObsVar":
                result = obs[terminal["obs_index"]]

            # Constants
            elif terminal["token"] == "Constant":  # constants
                if terminal["type"] == "Float":  # floats
                    result = float(p)

        # Functions
        else:
            fname = p[0]
            args = [self.eval(p[i+1], obs) for i in range(self.F[fname]["arity"])]

            # IFLTE
            if fname == "IFLTE":
                result = IFLTE(args)

        return result
