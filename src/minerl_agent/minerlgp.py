class MineRLGP:
	"""
	A GP algorithm for the MineRL Navigate Dense environment.
	"""

	# TODO:
	# Finish implementing __init__ to create the initial population. 

	# Parameters
	_pop_size = 4

	# Function and terminal sets
	_fset = []
	_tset = ['', 'forward']

	_init_pop = []  # initial population

	def __init__(self, pop_size=4):
		"""Creates an initial population."""

		self._pop_size = pop_size
		print("Created GP population with size " + str(self._pop_size))
