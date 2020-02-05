import gym
import numpy as np

class GPAgent:
    """
    GP template class. Implements all the functionality of a GP agent 
    for a gym environment and is parameterised by a data object with
    the parameters required to train an agent on a specific environment. 
    """

    def __init__(self, info):
        self.env_name = info["env_name"]
        
        # Program structure params
        self.p_type = info["program_type"]
        self.T = info["T"]
        self.F = info["F"]
        self.max_d = info["max_depth"]
        self.growth_rate = info["term_growth_rate"]  # probability of growing a terminal instead of a function
        
        # GP experiment params
        self.n = info["pop_size"]
        self.num_eps = info["num_eps"]
        self.max_gens = info["max_gens"]
        self.term_fit = info["term_fit"]

    def run(self, seed, render=True):
        best_program = self._train()
        print(best_program)

        env = gym.make(self.env_name)
        env.seed(seed)

        net_reward = self._run_eps(best_program, env, self.num_eps, render=render)
        print("Reward: {}".format(net_reward / self.num_eps))

        env.close()

    def _train(self):
        best_program = []

        # Generate initial population
        init_pop = [self._gen_prog(self.max_d, 'grow', self.p_type) for _ in range(self.n)]

        # Evolution loop
        current_pop = init_pop
        for gen_idx in range(self.max_gens):
            print("Generation {}".format(gen_idx+1))
            
            # Print the first 10 programs
            for i in range(self.n):
                if i < 10:
                    print(current_pop[i])
                else:
                    break
            
            # Evaluate population fitness
            fit_scores = self._fit(current_pop)
            print("Mean fitness = {}\n".format(sum(fit_scores)/len(fit_scores)))

            # If fitness termination criterion is met, select best program and stop evolution
            max_fitness = max(fit_scores)
            if max_fitness >= self.term_fit:
                best_program = current_pop[fit_scores.index(max_fitness)]
                break

            # If this is the last generation, pick the best individual
            if gen_idx >= self.max_gens-1:
                best_program_idx = fit_scores.index(max(fit_scores))
                best_program = current_pop[best_program_idx]

            # Selection & reproduction
            else:
                next_pop = [self._select(current_pop, fit_scores) for _ in range(self.n)]
                current_pop = next_pop

        return best_program

"""
    def _gen_prog(self, max_d, method, type):
        """
        Generates a random program with a fixed max depth using the terminal and function sets. 
        Supported methods: full and growth.

        max_d: maximum program depth
        method: "grow" | "full"
        """
        
        prog = None

        # Filter functions and terminals to only include items of the specified type.
        filt_terms = list(dict(filter(lambda term: term[1]["type"]==type, self.T.items())).keys())
        filt_funcs = list(dict(filter(lambda func: func[1]["type"]==type, self.F.items())).keys())

        if max_d == 0 or (method == "grow" and self.growth_rate > np.random.rand()):
            prog = np.random.choice(filt_terms)
        else:
            if filt_funcs:
                # Generate function of correct arity and arg type
                func = np.random.choice(filt_funcs)
                arg_types = self.F[func]["arg_types"]
                args = [self._gen_prog(max_d-1, method, t) for t in arg_types]
                prog = [func] + args
            else:  # a function of the required type doesn't exist
                prog = np.random.choice(filt_terms)

        return prog
"""

    def _fit(self, pop):
        scores = []

        env = gym.make(self.env_name)
        for p in pop:
            # Run episodes and store average reward
            net_reward = self._run_eps(p, env, self.num_eps)            
            scores.append(net_reward / self.num_eps)
        
        env.close()
        return scores

    def _run_eps(self, p, env, num_eps, render=False):
        net_reward = 0

        # Run episodes
        for _ in range(num_eps):
            
            if self.env_name == "Pendulum-v0":
                ep_reward = self._run_ep_pendulum(p, env, render=render)
            else:
                ep_reward = self._run_ep(p, env, render=render)
            
            net_reward += ep_reward
        
        return net_reward

    def _run_ep(self, p, env, render=False):
        reward = 0

        done = False
        obs = env.reset()

        while not done:
            if render:
                env.render()

            action = self._eval(p, obs)
            obs, rew, done, _ = env.step(action)
            reward += rew

        return reward

    def _run_ep_pendulum(self, p, env, render=False):
        reward = -1

        obs = env.reset()
        for _ in range(2000):
            if render:
                env.render()

            action = self._eval(p, obs)
            obs, rew, _, _ = env.step(action)
            reward += rew

        return reward

    def _select(self, pop, fit_scores):
        """
        Fitness Proportionate Selection (Roulette Wheel Selection)
        """

        selected = None

        # Turn fitness scores to the inverse of their absolute value.
        # E.g. fit_i = 1/abs(fit_i)
        # This makes roulette wheel selection work for negative fitness scores (e.g. for Pendulum-v0).
        if (fit_scores[0] < 0):
            fit_scores = [1/abs(score) for score in fit_scores]

        F = sum(fit_scores)
        r = np.random.uniform(0, F)

        acc = 0
        for i in range(len(fit_scores)):
            acc += fit_scores[i]

            if acc > r:
                selected = pop[i]
                break

        return selected

    def _eval(self, p, obs):
        action = 0

        # Terminals
        if type(p) is not list:
            term = self.T[p]
            if term["token"] == "StateVar":  # state variable
                action = obs[term["state_index"]]
            elif term["token"] == "Constant":  # constant
                if term["type"] == "Float":  # float
                    action = float(p)
            elif term["type"] == "Action":  # action
                action = int(p)

        # Functions
        else:
            fname = p[0]
            args = [self._eval(p[i+1], obs) for i in range(self.F[fname]["arity"])]
            if fname == "IFLTE":
                action = self._IFLTE(args)

        # Actions in Pendulum-v0 are arrays with a single float in them.
        if (self.env_name == "Pendulum-v0"):
            action = [action]

        return action

    def _IFLTE(self, args):
        action = 0

        if args[0] <= args[1]:
            action = args[2]
        else:
            action = args[3]

        return action
