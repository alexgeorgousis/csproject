import gym
import numpy as np

class GPAgent:
    """
    GP template class. Implements all the functionality of a GP agent 
    for a gym environment and is parameterised by a data object with
    the parameters for a specific environment. 
    """

    def __init__(self, info):
        self.env_name = info["env_name"]
        self.T = info["T"]
        self.F = info["F"]
        self.max_d = info["max_depth"]
        self.n = info["pop_size"]
        self.num_eps = info["num_eps"]
        self.max_gens = info["max_gens"]
        self.term_fit = info["term_fit"]

    def run(self):
        best_program = self._train()
        
        env = gym.make(self.env_name)
        obs = env.reset()
        net_reward = 0
        done = False

        while not done:
            env.render()
            action = self._eval(best_program, obs)
            obs, rew, done, _ = env.step(action)
            net_reward += rew
        
        print("Net reward: {}".format(net_reward))
        env.close()

    def _train(self):
        best_program = []

        # Generate initial population
        init_pop = [self._gen_prog(self.max_d, 'grow') for _ in range(self.n)]
        for p in init_pop:
            print(p)

        # Evolution loop

        return best_program

    def _eval(self, p, obs):
        return 0

    def _gen_prog(self, max_d, method):
        """
        Generates a random program with a fixed max depth using the terminal and function sets. 
        Supported methods: full and growth.

        max_d: maximum program depth
        method: "grow" | "full"
        """
        
        prog = None

        len_T = len(self.T)
        len_F = len(self.F)
        p_term = len_T / (len_T + len_F)  # probability of choosing a terminal

        # Pick random terminal
        if max_d == 0 or (method == "grow" and p_term > np.random.rand()):
            prog = np.random.choice(self.T)
        else:
            # Pick random function and recursively generate random arguments for it
            func = np.random.choice(list(self.F.keys()))
            arity = self.F[func]["arity"]
            args = [self._gen_prog(max_d-1, method) for _ in range(arity)]
            prog = [func] + args

        return prog
