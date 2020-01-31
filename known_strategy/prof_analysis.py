from pstats import Stats, SortKey


# Train profile
p = Stats('train.prof')
p.strip_dirs()
p.sort_stats(SortKey.CUMULATIVE)
p.print_stats('agent.py:|cartpole.py:')
