import gym

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
        self.program_depth = info["program_depth"]
        self.pop_size = info["pop_size"]
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
        return ['IFLTE', '0', '0', '1', '0']

    def _eval(self, p, obs):
        return 0
