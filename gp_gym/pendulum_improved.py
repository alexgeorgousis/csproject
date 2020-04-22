import gym
import numpy as np
import time
from deap import gp, base, creator, tools
import copy
import operator


class PendulumImproved:
    """
    This class implements a GP agent for the Pendulum-v0 gym environment.
    """

    def __init__(self, info):

        # The number of states the pendulum env is split into (and therefore the number of sub-programs to generate for each individual)
        self.NUM_STATES = 4 

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
        # self.pset.addPrimitive(operator.add, 2)
        # self.pset.addPrimitive(operator.sub, 2)
        self.pset.addPrimitive(operator.mul, 2)
        self.pset.addPrimitive(operator.neg, 1)

        self.pset.addTerminal(0.0)
        self.pset.addTerminal(0.25)
        self.pset.addTerminal(0.5)
        self.pset.addTerminal(1.0)

        # Program generation functions
        creator.create("CostMin", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.CostMin, pset=self.pset)
        self.toolbox = base.Toolbox()
        method = gp.genGrow if info["method"] == "grow" else gp.genFull
        self.toolbox.register("gen_exp", method, pset=self.pset, min_=0, max_=info["max_depth"])
        self.toolbox.register("gen_program", tools.initIterate, creator.Individual, self.toolbox.gen_exp)
        self.toolbox.register("gen_pop", tools.initRepeat, list, self.toolbox.gen_program)

        # Fitness function
        self.toolbox.register("fit", self.fit)

        # Genetic operators
        self.toolbox.register("clone", self._clone)
        self.toolbox.register("select", self._tournament_sel, tournsize=self.tour_size)
        self.toolbox.register("mut_gen_exp", method, pset=self.pset, min_=0, max_=info["max_depth"])
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.mut_gen_exp, pset=self.pset)

        # Statistics functions
        self.stats = tools.Statistics(key=lambda indiv: indiv[0].fitness.values)
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
        pop = [self.toolbox.gen_pop(n=self.NUM_STATES) for _ in range(self.pop_size)]

        # Evolution loop
        gen_count = 1
        while (gen_count <= self.max_gens) and (not best_program):

            print("\nGeneration {}...".format(gen_count))
            start = time.time()

            # Evaluate population fitness
            pop_fitnesses = [self.toolbox.fit(p, self.num_eps, self.num_steps) for p in pop]
            for indiv, fitness in zip(pop, pop_fitnesses):
                for p in indiv:
                    p.fitness.values = fitness

            # Record population statistics
            record = self.stats.compile(pop)
            self.logbook.record(gen=gen_count, **record)

            # Check termination criteria
            max_fitness = max(pop_fitnesses)[0]
            if (gen_count >= self.max_gens) or (max_fitness >= self.term_fit):
                # Find the first individual in the population with the maximum fitness
                best_program = pop[pop_fitnesses.index(max_fitness)]

                # If the terminal fitness was reached, let me know!
                if max_fitness >= self.term_fit:
                    print("\nFITNESS {} found!!!".format(max_fitness))
                    print([str(p) for p in best_program])
                
            else:
                # Apply selection and reset fitness values
                selected = self.toolbox.select(pop, pop_fitnesses, self.pop_size)

                for indiv in selected:
                    for p in indiv:
                        del p.fitness.values

                # Apply mutation
                for indiv in selected:
                    for p in indiv:
                        if np.random.rand() < self.mut_rate:
                            p = self.toolbox.mutate(p)[0]
                            del p.fitness.values

                # Update population
                pop = selected

            gen_count += 1

            end = time.time()
            print("Train time: {}".format(end-start))

        return best_program


    def fit(self, indiv, num_eps, num_steps, render=False):
        fitness = 0.0
        net_cost = 0.0

        # Create a program template that splits the Pendulum environment
        # in 4 quadrants and injects the programs in the blanks.
        template = "IFLTE(sintheta, 0.0,\
                          IFLTE(costheta, 0.0, {}, {}),\
                          IFLTE(costheta, 0.0, {}, {}))".format(*indiv)

        executable = gp.compile(template, self.pset)

        # Run individual on the environment to compute its fitness
        env = gym.make(self.env_name)
        for _ in range(num_eps):
            obs = env.reset()
            for _ in range(num_steps):
                
                if render:
                    env.render()
                    time.sleep(0.02)
                
                action = executable(obs[0], obs[1], obs[2])
                obs, cost, _, _ = env.step([action])
                net_cost += cost

        env.close()
        fitness = net_cost/num_eps
        return np.round(fitness, decimals=5),

    def IFLTE(self, arg1, arg2, arg3, arg4):
        if arg1 <= arg2:
            return arg3
        else:
            return arg4

    def _clone(self, indiv):
        clone = [copy.deepcopy(p) for p in indiv]
        return clone

    def _tournament_sel(self, pop, pop_fitnesses, k, tournsize=3):
        """
        Modified tournament selection to work on populations of 
        individuals that are lists of programs (instead of being programs themselves).

        :param pop: population to select individuals from
        :param pop_fitnesses: list containing the fitness of each individual in the population (parallel to pop)
        :param k: number of individuals to select
        :param tournsize: tournament size
        :return: list of (cloned) selected individuals 
        """

        selected = []
        for _ in range(k):
            cand_indices = [np.random.choice(range(len(pop))) for _ in range(tournsize)]
            cand_fitnesses = [pop_fitnesses[i] for i in cand_indices]
            max_fit_idx = cand_fitnesses.index(max(cand_fitnesses))
            best_cand = pop[cand_indices[max_fit_idx]]
            selected.append(self._clone(best_cand))
        return selected
