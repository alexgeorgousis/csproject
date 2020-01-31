
def search_grid(min_size, max_size, size_incr):
	"""
	min_size - size of inner-most nxn grid
	min_size - size of outer-most nxn grid
	size_incr - size increment
	"""
	
	grid = []
	
	# Actions
	forward = {'forward': 1, 'jump': 1}
	turn = {'camera': [0, -90]}

	# Each iteration defines actions for 1 grid
	for size in range(min_size, max_size+size_incr, size_incr):
		
		# First side (forward)
		grid.append(forward)
		grid.append(turn)

		# Second side (left)
		for _ in range(size-2):
			grid.append(forward)
		grid.append(turn)

		# Third side (back)
		# Fourth side (right)
		for _ in range(2):
			for _ in range(size-1):
				grid.append(forward)
			grid.append(turn)

		# Fifth side (forward)
		for _ in range(size-1):
			grid.append(forward)

	return grid
