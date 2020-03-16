import gym
import numpy as np
import time
from deap import gp, base, creator, tools
import copy
import operator

class Action:
    def __init__(self, actions):
        self.a0 = actions[0]
        self.a1 = actions[1]
        self.a2 = actions[2]
        self.a3 = actions[3]

    def get_actions(self):
        return [self.a0, self.a1, self.a2, self.a3]

class BipedalWalker:
    """
    This class implements a GP agent for the Bipedal Walker gym environment.
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
        obs_types = [
            float, float, float, float, float, float,
            float, float, float, float, float, float,
            float, float, float, float, float, float,
            float, float, float, float, float, float]
        self.pset = gp.PrimitiveSetTyped("main", obs_types, Action)
        self.pset.renameArguments(
            ARG0="hull_angle", 
            ARG1="hul_angularVelocity",
            ARG2="vel_x",
            ARG3="vel_y",
            ARG4="hip_joint_1_angle",
            ARG5="hip_joint_1_speed",
            ARG6="knee_joint_1_angle",
            ARG7="knee_joint_1_speed",
            ARG8="leg_1_ground_contact_flag",
            ARG9="hip_joint_2_angle",
            ARG10="hip_joint_2_speed",
            ARG11="knee_joint_2_angle",
            ARG12="knee_joint_2_speed",
            ARG13="leg_2_ground_contact_flag",
            ARG14="lidar_reading_1",
            ARG15="lidar_reading_2",
            ARG16="lidar_reading_3",
            ARG17="lidar_reading_4",
            ARG18="lidar_reading_5",
            ARG19="lidar_reading_6",
            ARG20="lidar_reading_7",
            ARG21="lidar_reading_8",
            ARG22="lidar_reading_9",
            ARG23="lidar_reading_10")

        self.pset.addPrimitive(self.ACCUM, [float, float, float, float], Action)
        self.pset.addPrimitive(self.IFLTE, [float, float, float, float], float)
        # self.pset.addPrimitive(operator.add, 2)
        # self.pset.addPrimitive(operator.sub, 2)
        # self.pset.addPrimitive(operator.mul, 2)
        # self.pset.addPrimitive(operator.neg, 1)

        self.pset.addTerminal(0.0, float)
        self.pset.addTerminal(0.25, float)
        self.pset.addTerminal(0.5, float)
        self.pset.addTerminal(1.0, float)
        self.pset.addTerminal(0, float)
        self.pset.addTerminal(1, float)
        self.pset.addTerminal(
            Action(np.random.uniform(low=-1.0, high=1.0, size=4)), 
            Action, 
            name="random_action")

        # Program generation functions
        creator.create("FitMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitMax, pset=self.pset)
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
        net_reward = 0.0

        env = gym.make(self.env_name)
        executable = gp.compile(indiv, self.pset)

        for _ in range(num_eps):
            obs = env.reset()
            done = False
            while not done:
                
                if render:
                    env.render()
                    # time.sleep(0.02)
                
                action_obj = executable(
                    obs[0],  obs[1],  obs[2],  obs[3],  obs[4],  obs[5],
                    obs[6],  obs[7],  obs[8],  obs[9],  obs[10], obs[11],
                    obs[12], obs[13], obs[14], obs[15], obs[16], obs[17],
                    obs[18], obs[19], obs[20], obs[21], obs[22], obs[23]
                )

                action = action_obj.get_actions()
                obs, reward, done, _ = env.step(action)
                net_reward += reward

        env.close()
        fitness = net_reward/num_eps
        return np.round(fitness, decimals=5),

    def _clone(self, indiv):
        return copy.deepcopy(indiv)

    def IFLTE(self, arg1, arg2, arg3, arg4):
        if arg1 <= arg2:
            return arg3
        else:
            return arg4

    def ACCUM(self, arg1, arg2, arg3, arg4):
        return Action([arg1, arg2, arg3, arg4])
