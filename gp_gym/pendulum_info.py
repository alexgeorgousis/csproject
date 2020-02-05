"""
Data object representation of GP agent 
for the Pendulum-v0 environment.
"""

info = {
    "env_name": "Pendulum-v0",
    "program_type": "Float",
    "T": {
        "costheta": {"type":"Float", "token": "ObsVar", "obs_index": 0},
        "sintheta": {"type":"Float", "token": "ObsVar", "obs_index": 1},
        "thetadot": {"type":"Float", "token": "ObsVar", "obs_index": 2},
        "0.0": {"type":"Float", "token": "Constant"},
        "1.0": {"type":"Float", "token": "Constant"},
        "-1.0": {"type":"Float", "token": "Constant"},
    },
    "F": {"IFLTE": {"arity": 4, "type": "Float", "arg_types": ["Float", "Float", "Float", "Float"]}},
    "max_depth": 2,
    "term_growth_rate": 1.0,
    "method": 'grow',
    "pop_size": 1,
    "num_eps": 100,
    "max_gens": 1,
    "term_fit": -200.0
}
