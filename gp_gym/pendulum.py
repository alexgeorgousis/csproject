from gp_gym import *


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
        self.num_eps = info["num_eps"]
        self.max_gens = info["max_gens"]
        self.term_fit = info["term_fit"]


    def batch_fit(self, pop, num_time_steps):
        """
        Computes the fitness score of every program in a population.

        pop: population of programs
        num_eps: number of episodes to evaluate each program on
        """

        env = gym.make(self.env_name)
        scores = [self.fit(env, p, num_time_steps) for p in pop]
        env.close()
        return scores


    def fit(self, env, p, num_time_steps):
        """
        Computes the fitness score (total reward) of a program in a single Pendulum-v0 episode.
        
        The reward in Pendulum-v0 is always negative. To make it work with fitness proportionate selection, 
        this function converts it to a positive number, while preserving its relative fitness information:
        reward = 1/|reward|

        env: gym environment object
        p: program to evaluate
        num_eps: number of time steps to run the episode for
        return: fitness score (float)
        """

        score = 0.0

        obs = env.reset()
        for _ in range(num_time_steps):
            action = self.eval(p, obs)
            obs, reward, done, info = env.step([action])
            score += reward

        # Make score positive
        score = 1 / abs(score)

        return score


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
