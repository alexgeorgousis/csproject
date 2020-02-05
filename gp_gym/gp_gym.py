"""
The purpose of this module is to encapsulate common Genetic Programming and OpenAI gym functions
to make them reusable in different environments.
"""

import gym
import numpy as np


def gen_init_pop(pop_size, T, F, max_depth, method, t_rate, p_type):
    return [gen_program(T, F, max_depth, method, t_rate, p_type) for _ in range(pop_size)]


def gen_program(T, F, max_depth, method, t_rate, p_type):
    """
    Generates a random program with a fixed max depth using the terminal and function sets. 
        Supported methods: full and growth.

        T: terminal set
        F: function set
        max_depth: maximum program depth
        method: "grow" | "full"
        t_rate: probability of generating a terminal (only used if method='grow')
        p_type: the Type of the program to generate
        return: a literal or a list that represents a program
    """

    p = None

    # Filter terminals to only include items of the specified type.
    filt_terms = list(dict(filter(lambda term: term[1]["type"]==p_type, T.items())).keys())

    # Check if function set is empty
    filt_funcs = []
    if len(F) > 0:
        filt_funcs = list(dict(filter(lambda func: func[1]["type"]==p_type, F.items())).keys())

    # Pick a random terminal or function
    if max_depth == 0 or (method == "grow" and t_rate > np.random.rand()):
        p = np.random.choice(filt_terms)
    else:
        if filt_funcs:
            # Generate function of correct arity and arg type
            func = np.random.choice(filt_funcs)
            arg_types = F[func]["arg_types"]
            args = [gen_program(T, F, max_depth-1, method, t_rate, arg_type) for arg_type in arg_types]
            p = [func] + args
        else:  # a function of the required type doesn't exist
            p = np.random.choice(filt_terms)

    return p



