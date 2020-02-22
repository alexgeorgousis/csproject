"""
Data object representation of GP agent 
for the Pendulum-v0 environment.
"""

info = {
    "env_name": "Pendulum-v0",
    "max_depth": 2,
    "term_growth_rate": 1.0,
    "method": 'grow',
    "pop_size": 1,
    "num_time_steps": 200,
    "num_eps": 100,
    "max_gens": 1,
    "term_fit": -800.0,
    "mutation_rate": 0.1,
    "tournament_size": 1
}