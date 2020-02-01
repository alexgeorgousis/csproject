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
        self.T = info["T"]
        self.F = info["F"]
        self.max_d = info["max_depth"]
        self.growth_rate = info["term_growth_rate"]  # probability of growing a terminal instead of a function
        
        # GP experiment params
        self.n = info["pop_size"]
        self.num_eps = info["num_eps"]
        self.max_gens = info["max_gens"]
        self.term_fit = info["term_fit"]

    def run(self):
        best_program = self._train()
        print(best_program)

        env = gym.make(self.env_name)
        net_reward = 0

        # Run episodes
        for _ in range(self.num_eps):
            ep_reward = 0
            done = False
            obs = env.reset()

            # Run single episode
            while not done:
                action = self._eval(best_program, obs)
                obs, rew, done, _ = env.step(action)
                ep_reward += rew
                
            net_reward += ep_reward
        
        print("Reward: {}".format(net_reward / self.num_eps))
        env.close()

    def _train(self):
        best_program = []

        # Generate initial population
        init_pop = [self._gen_prog(self.max_d, 'full') for _ in range(self.n)]

        # Evolution loop
        current_pop = init_pop
        for gen_idx in range(self.max_gens):
            print("Generation {}".format(gen_idx))
            print(current_pop)
            
            # Evaluate population fitness
            fit_scores = self._fit(current_pop)

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

    def _gen_prog(self, max_d, method, type="Action"):
        """
        Generates a random program with a fixed max depth using the terminal and function sets. 
        Supported methods: full and growth.

        max_d: maximum program depth
        method: "grow" | "full"
        """
        
        prog = None

        # Filter functions and terminals to only include items
        # of the type specified in the arguments.
        filt_terms = list(dict(filter(lambda term: term[1]["type"]==type, self.T.items())).keys())
        filt_funcs = list(dict(filter(lambda func: func[1]["type"]==type, self.F.items())).keys())

        if max_d == 0 or (method == "grow" and self.growth_rate > np.random.rand()):
            prog = np.random.choice(filt_terms)
        else:
            if filt_funcs:
                # Generate function of correct arity and arg type
                func = np.random.choice(filt_funcs)
                arg_types = self.F[func]["arg_types"]
                args = [self._gen_prog(max_d-1, method, type=t) for t in arg_types]
                prog = [func] + args
            else:  # a function of the required type doesn't exist
                prog = np.random.choice(filt_terms)

        return prog

    def _fit(self, pop):
        scores = []

        env = gym.make(self.env_name)
        for p in pop:
            net_reward = 0

            # Run episodes
            for _ in range(self.num_eps):
                ep_reward = 0
                done = False
                obs = env.reset()

                # Run single episode
                while not done:
                    action = self._eval(p, obs)
                    obs, rew, done, _ = env.step(action)
                    ep_reward += rew
                    
                net_reward += ep_reward

            # Store average reward
            scores.append(net_reward / self.num_eps)
        
        env.close()
        return scores

    def _select(self, pop, fit_scores):
        """
        Fitness Proportionate Selection (Roulette Wheel Selection)
        """

        selected = None

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
                return obs[term["state_index"]]
            elif term["token"] == "Constant":  # constant
                if term["type"] == "Float":  # float
                    return float(p)
            elif term["type"] == "Action":  # action
                return int(p)

        # Functions
        else:
            fname = p[0]
            args = [self._eval(p[i+1], obs) for i in range(self.F[fname]["arity"])]
            if fname == "IFLTE":
                action = self._IFLTE(args)

        return action

    def _IFLTE(self, args):
        action = 0

        if args[0] <= args[1]:
            action = args[2]
        else:
            action = args[3]

        return action
