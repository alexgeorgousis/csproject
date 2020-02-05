"""
Data object representation of GP agent 
for the Cartpole-v0 environment.
"""

info = {
    "env_name": "CartPole-v0",
    "program_type": "Action",
    "T": {
        "pa": {"type":"Float", "token": "StateVar", "state_index": 2}, 
        "pv": {"type":"Float", "token": "StateVar", "state_index": 3}, 
        "0.0": {"type":"Float", "token": "Constant"}, 
        "0.025": {"type":"Float", "token": "Constant"}, 
        "0": {"type":"Action", "token": ""}, 
        "1": {"type":"Action", "token": ""}
    },
    "F": {"IFLTE": {"arity": 4, "type": "Action", "arg_types": ["Float", "Float", "Action", "Action"]}},
    "max_depth": 2,
    "term_growth_rate": 1.0,
    "method": 'grow',
    "pop_size": 1,
    "num_eps": 100,
    "max_gens": 1,
    "term_fit": 195.0
}
