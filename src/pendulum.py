import gym
import numpy as np
import time
from deap import gp, base, creator, tools
import copy
import operator


class Pendulum:
    """
    This class implements a GP agent for the Pendulum-v0 gym environment.
    """

    def __init__(self, info):

        # Extract agent info
        self.env_name  = info["env_name"]
        self.pop_size  = info["pop_size"]
        self.max_gens  = info["max_gens"]
        self.num_eps   = info["num_eps"]
        self.num_steps = info["num_time_steps"]
        self.term_fit  = info["term_fit"]
        self.mut_rate  = info["mutation_rate"]
        self.tour_size = info["tournament_size"]

        # Primitive set
        self.pset = gp.PrimitiveSet("main", 3)
        self.pset.renameArguments(ARG0="costheta", ARG1="sintheta", ARG2="thetadot")

        self.pset.addPrimitive(self.IFLTE, 4)
        self.pset.addPrimitive(operator.neg, 1)

        self.pset.addTerminal(0.0)
        self.pset.addTerminal(0.25)

        # Program generation functions
        creator.create("CostMin", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.CostMin, pset=self.pset)
        self.toolbox = base.Toolbox()
        method = gp.genGrow if info["method"] == "grow" else gp.genFull
        self.toolbox.register("gen_exp", method, pset=self.pset, min_=1, max_=info["max_depth"])
        self.toolbox.register("gen_program", tools.initIterate, creator.Individual, self.toolbox.gen_exp)
        self.toolbox.register("gen_pop", tools.initRepeat, list, self.toolbox.gen_program)

        # Fitness evaluation function
        self.toolbox.register("fit", self.fit)

        # Genetic operators
        self.toolbox.register("clone", self._clone)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tour_size)
        self.toolbox.register("mut_gen_exp", method, pset=self.pset, min_=0, max_=info["max_depth"])
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.mut_gen_exp, pset=self.pset)

        # Statistics functions
        self.stats = tools.Statistics(key=lambda indiv: indiv.fitness.values)
        self.stats.register("avg", np.mean)
        self.logbook = tools.Logbook()


    def train(self):
        """
        This is where the GP algorithm is implemented.
        This method uses all the auxiliary methods to perform a full GP run.

        :returns: the best program found during training
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
            pop_fitness = [self.toolbox.fit(p, self.num_eps, self.num_steps) for p in pop]
            for indiv, fitness in zip(pop, pop_fitness):
                indiv.fitness.values = fitness

            # Record population statistics
            record = self.stats.compile(pop)
            self.logbook.record(gen=gen_count, **record)

            # Check termination criteria
            max_fitness = max(pop_fitness)[0]
            if (gen_count >= self.max_gens) or (max_fitness >= self.term_fit):
                for indiv in pop:
                    if indiv.fitness.values[0] >= max_fitness:
                        best_program = indiv
                        break
                
            else:
                # Apply selection
                selected = self.toolbox.select(pop, self.pop_size)

                # Clone individuals to avoid reference issues
                # and reset their fitness values
                selected = [self.toolbox.clone(indiv) for indiv in selected]
                for indiv in selected:
                    del indiv.fitness.values

                # Apply mutation
                for indiv in selected:
                    if np.random.rand() < self.mut_rate:
                        indiv = self.toolbox.mutate(indiv)[0]
                        del indiv.fitness.values

                # Update population
                pop = selected

            gen_count += 1

            end = time.time()
            print("Train time: {}".format(end-start))

        return best_program


    def fit(self, indiv, num_eps, num_steps, render=False):
        fitness = 0.0
        ep_cost = 0.0
        ep_cost_lst = []

        env = gym.make(self.env_name)
        executable = gp.compile(indiv, self.pset)

        for _ in range(num_eps):
            obs = env.reset()
            ep_cost = 0.0
            for _ in range(num_steps):
                
                if render:
                    env.render()
                    time.sleep(0.02)
                
                action = executable(obs[0], obs[1], obs[2])
                obs, cost, _, _ = env.step([action])
                ep_cost += cost
            ep_cost_lst.append(ep_cost)

        env.close()
        fitness = np.mean(ep_cost_lst)
        return np.round(fitness, decimals=5),

    def IFLTE(self, arg1, arg2, arg3, arg4):
        if arg1 <= arg2:
            return arg3
        else:
            return arg4

    def _clone(self, indiv):
        return copy.deepcopy(indiv)
