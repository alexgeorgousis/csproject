actions = []

for i in range(3, 10, 2):
	actions.append(i)

	actions.append({'forward': 1})
	actions.append({'camera': [0, -90]})

	for _ in range(i-2):
		actions.append({'forward': 1})
	actions.append({'camera': [0, -90]})	

	for _ in range(i-1):
		actions.append({'forward': 1})
	actions.append({'camera': [0, -90]})

	for _ in range(i-1):
		actions.append({'forward': 1})
	actions.append({'camera': [0, -90]})

	for _ in range(i-1):
		actions.append({'forward': 1})

for a in actions:
	print(a)
