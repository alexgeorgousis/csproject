from gp_gym import *


class CartPole:
    """
    This class implements a GP agent for the CartPole-v0 gym environment.
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


    def batch_fit(self, pop, num_eps):
        """
        Computes the average fitness score (over a specified number of episodes) 
        of every program in a population.

        pop: population of programs
        num_eps: number of episodes to evaluate each program on
        """

        env = gym.make(self.env_name)
        scores = [self.fit(env, p, num_eps) for p in pop]
        env.close()
        return scores


    def fit(self, env, p, num_eps):
        """
        Computes the average fitness score of a program over a 
        specified number of episodes.

        env: gym environment object
        p: program to evaluate
        num_eps: number of episodes to run the program for
        return: fitness score (float)
        """

        score = 0.0

        for _ in range(num_eps):
            score += run_ep_while_not_done(env, p, self.eval)

        return score/num_eps


    def eval(self, p, obs):
        result = 0

        # Terminals
        if type(p) is not list:
            terminal = self.T[p]

            # Actions
            if terminal["type"] == "Action":
                result = int(p)
            
            # Observation variables
            elif terminal["token"] == "ObsVar":
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
                result = self.IFLTE(args)

        return result


    def IFLTE(self, args):
        """ Implements the IFLTE function. """
        return args[2] if args[0] <= args[1] else args[3]
