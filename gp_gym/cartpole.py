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
