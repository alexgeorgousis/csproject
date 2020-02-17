import gym
import numpy as np
import time
from deap import gp, base, creator, tools


class Pendulum:
    """
    This class implements a GP agent for the Pendulum-v0 gym environment.
    """

    def __init__(self, info):
        self.env_name = info["env_name"]
        
        # Primitive set
        self.pset = gp.PrimitiveSet("main", 3)
        self.pset.renameArguments(ARG0="costheta", ARG1="sintheta", ARG2="thetadot")
        
        self.pset.addPrimitive(self.IFLTE, 4)
        self.pset.addTerminal(-1.0)
        self.pset.addTerminal(0.0)
        self.pset.addTerminal(1.0)

        # Generation functions
        creator.create("CostMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.CostMin, pset=self.pset)

        self.toolbox = base.Toolbox()
        method = gp.genGrow if info["method"] == "grow" else gp.genFull
        self.toolbox.register("gen_exp", method, pset=self.pset, min_=1, max_=info["max_depth"])
        self.toolbox.register("gen_program", tools.initIterate, creator.Individual, toolbox.gen_exp)
        self.toolbox.register("gen_pop", tools.initRepeat, list, toolbox.gen_program)


    def fit(self, indiv):
        fitness = 0.0
        env = gym.make(self.env_name)

        net_cost = 0.0
        num_eps = 100
        for _ in range(num_eps):
            obs = env.reset()
            done = False
            while not done:
                # env.render()
                executable = gp.compile(indiv, self.pset)
                action = executable(obs[0], obs[1], obs[2])
                obs, cost, done, _ = env.step([action])
                net_cost += cost

        env.close()
        fitness = net_cost/num_eps
        return fitness

    def IFLTE(self, arg1, arg2, arg3, arg4):
        if arg1 <= arg2:
            return arg3
        else:
            return arg4
