import gym
import numpy as np
import time
from deap import gp, base, creator, tools


class Pendulum:
    """
    This class implements a GP agent for the Pendulum-v0 gym environment.
    """

    def __init__(self, info):

        # Extract agent info
        self.env_name = info["env_name"]
        self.pop_size = info["pop_size"]
        self.max_gens = info["max_gens"]
        self.term_fit = info["term_fit"]

        # Primitive set
        self.pset = gp.PrimitiveSet("main", 3)
        self.pset.renameArguments(ARG0="costheta", ARG1="sintheta", ARG2="thetadot")
        self.pset.addPrimitive(self.IFLTE, 4)
        self.pset.addTerminal(-1.0)
        self.pset.addTerminal(0.0)
        self.pset.addTerminal(1.0)

        # Program generation functions
        creator.create("CostMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.CostMin, pset=self.pset)
        self.toolbox = base.Toolbox()
        method = gp.genGrow if info["method"] == "grow" else gp.genFull
        self.toolbox.register("gen_exp", method, pset=self.pset, min_=1, max_=info["max_depth"])
        self.toolbox.register("gen_program", tools.initIterate, creator.Individual, self.toolbox.gen_exp)
        self.toolbox.register("gen_pop", tools.initRepeat, list, self.toolbox.gen_program)

        # Genetic operators
        self.toolbox.register("select", tools.selRoulette)


    def train(self):
        """
        This is where the GP algorithm is implemented.
        This method uses all the auxiliary methods to perform a full GP run.
        """

        best_program = None

        # Generate initial population
        pop = self.toolbox.gen_pop(n=self.pop_size)

        # Evolution loop
        gen_count = 1
        while (gen_count <= self.max_gens) and (not best_program):

            print("\nGeneration {}...".format(gen_count))
            start = time.time()

            # Evaluate population fitness
            pop_fitness = [self.fit(p) for p in pop]
            for indiv, fitness in zip(pop, pop_fitness):
                indiv.fitness.values = fitness

            # Check termination criteria
            max_fitness = max(pop_fitness)[0]
            if (gen_count >= self.max_gens) or (max_fitness >= self.term_fit):
                for indiv in pop:
                    if indiv.fitness.values >= max_fitness:
                        best_program = indiv
                        break
                
            else:
                # Apply selection & reproduction
                pop[:] = self.toolbox.select(pop, self.pop_size)

                # Apply mutation

            gen_count += 1

            end = time.time()
            print("Train time: {}".format(end-start))

        return best_program


    def fit(self, indiv):
        fitness = 0.0

        env = gym.make(self.env_name)
        executable = gp.compile(indiv, self.pset)

        net_cost = 0.0
        num_eps = 100
        for _ in range(num_eps):
            obs = env.reset()
            done = False
            while not done:
                action = executable(obs[0], obs[1], obs[2])
                obs, cost, done, _ = env.step([action])
                net_cost += cost

        env.close()
        fitness = net_cost/num_eps
        return np.round(fitness, decimals=5),

    def IFLTE(self, arg1, arg2, arg3, arg4):
        if arg1 <= arg2:
            return arg3
        else:
            return arg4
