

def is_all(x):
	# Filter "*" = all subsets [used in selecting subsets]
	return x is None or x == "*" or x == ["*"]