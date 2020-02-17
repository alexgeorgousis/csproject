"""
Data object representation of GP agent 
for the Pendulum-v0 environment.
"""

info = {
    "T": {
        "costheta": {"type":"Float", "token": "ObsVar", "obs_index": 0},
        "sintheta": {"type":"Float", "token": "ObsVar", "obs_index": 1},
        "thetadot": {"type":"Float", "token": "ObsVar", "obs_index": 2},
        "0.0": {"type":"Float", "token": "Constant"},
        "1.0": {"type":"Float", "token": "Constant"},
        "-1.0": {"type":"Float", "token": "Constant"},
    },
    "F": {"IFLTE": {"arity": 4, "type": "Float", "arg_types": ["Float", "Float", "Float", "Float"]}},

    "env_name": "Pendulum-v0",
    "max_depth": 2,
    "term_growth_rate": 1.0,
    "method": 'grow',
    "pop_size": 1,
    "num_time_steps": 200,
    "num_eps": 100,
    "max_gens": 1,
    "term_fit": -800.0,
    "mutation_rate": 1.0
}
