"""
Data object representation of GP agent 
for the Cartpole-v0 environment.
"""

info = {
    "env_name": "CartPole-v0",
    "T": {
        "pa": {"type":"Number"}, 
        "pv": {"type":"Number"}, 
        "0.0": {"type":"Number"}, 
        "0.025": {"type":"Number"}, 
        "0": {"type":"Action"}, 
        "1": {"type":"Action"}
    },
    "F": {"IFLTE": {"arity": 4}},
    "max_depth": 2,
    "term_growth_rate": 0.5,
    "pop_size": 1,
    "num_eps": 100,
    "max_gens": 1,
    "term_fit": 195.0
}
