

def branch(score, fulcrum, low, high):
	if score < fulcrum:
		return low
	else:
		return high