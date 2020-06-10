import gym
from gp_gym import gen_init_pop, select, run_ep_while_not_done, IFLTE


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


    def train(self):
        best_program = None

        # Generate initial population
        current_pop = gen_init_pop(self.pop_size, self.T, self.F, self.max_depth, self.method, self.t_rate, self.p_type)

        # Evolution loop
        gen_idx = 0
        while (not best_program) and (gen_idx < self.max_gens):

            # Evaluate population fitness
            fit_scores = self.batch_fit(current_pop, self.num_eps)
            max_fitness = max(fit_scores)

            # Check termination criteria
            if (max_fitness >= self.term_fit) or (gen_idx >= self.max_gens - 1):
                best_program = current_pop[fit_scores.index(max_fitness)]

            # Evolve next generation
            else:
                current_pop = [select(current_pop, fit_scores) for _ in range(self.pop_size)]
                gen_idx += 1

        return best_program


    def batch_fit(self, pop, num_eps, render=False):
        """
        Computes the average fitness score (over a specified number of episodes) 
        of every program in a population.

        pop: population of programs
        num_eps: number of episodes to evaluate each program on
        """

        env = gym.make(self.env_name)
        scores = [self.fit(p, num_eps, env=env, render=render) for p in pop]
        env.close()
        return scores


    def fit(self, p, num_eps, env=None, render=False):
        """
        Computes the average fitness score of a program over a 
        specified number of episodes.

        env: gym environment object
        p: program to evaluate
        num_eps: number of episodes to run the program for
        return: fitness score (float)
        """

        score = 0.0

        if not env:
            env = gym.make(self.env_name)

        for _ in range(num_eps):
            score += run_ep_while_not_done(env, p, self.eval, render=render)

        return score/num_eps


    def eval(self, p, obs):
        """
        Interprets a program and evaluates it to an action 
        given an observation from the environment.

        p: program to interpret
        obs: observation : [float]
        return: action {0, 1}
        """

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
                result = IFLTE(args)

        return result
