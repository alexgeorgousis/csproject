"""
Data object representation of GP agent 
for the Pendulum-v0 environment.
"""

info = {
    "env_name": "Pendulum-v0",
    "program_type": "Float"
    "T": {
        "costheta": {"type":"Float", "token": "StateVar", "state_index": 0},
        "sintheta": {"type":"Float", "token": "StateVar", "state_index": 1},
        "thetadot": {"type":"Float", "token": "StateVar", "state_index": 2},
        "-2.0": {"type":"Float", "token": "Constant"},
        "-1.0": {"type":"Float", "token": "Constant"},
        "0.0":  {"type":"Float", "token": "Constant"},
        "1.0":  {"type":"Float", "token": "Constant"},
        "2.0":  {"type":"Float", "token": "Constant"},
    },
    "F": {"IFLTE": {"arity": 4, "type": "Action", "arg_types": ["Float", "Float", "Action", "Action"]}},
    "max_depth": 2,
    "term_growth_rate": 1.0,
    "pop_size": 1,
    "num_eps": 100,
    "max_gens": 1,
    "term_fit": 195.0
}
