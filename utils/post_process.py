from numpy import ndarray, float32, zeros_like


def postprocess_synth_20(X: ndarray, p_threshold: float = 0.5, smooth_p: bool = False):
	"""
	X shape [N,T,20]
	layout:
	  [vx, vy,
	   LF_dx, LF_dy, LF_p,
	   LR_dx, LR_dy, LR_p,
	   RF_dx, RF_dy, RF_p,
	   RR_dx, RR_dy, RR_p,
	   F_dx,  F_dy,  F_p,
	   R_dx,  R_dy,  R_p]
	"""
	X = X.astype(float32, copy=True)

	nbr = X[:, :, 2:20].reshape(X.shape[0], X.shape[1], 6, 3)
	dx = nbr[..., 0]
	dy = nbr[..., 1]
	p = nbr[..., 2]

	if smooth_p and X.shape[1] >= 3:
		p2 = p.copy()
		p2[:, 1:-1, :] = (p[:, :-2, :] + p[:, 1:-1, :] + p[:, 2:, :]) / 3.0
		p[:] = p2

	present = p > p_threshold

	# hard projection of absence
	dx[~present] = 0.0
	dy[~present] = 0.0
	p[:] = present.astype(float32)

	# slot semantics:
	# front slots: LF, RF, F  if dy >= 0
	# rear  slots: LR, RR, R  if dy <= 0
	front_slots = [0, 2, 4]
	rear_slots = [1, 3, 5]

	for s in front_slots:
		bad = present[:, :, s] & (dy[:, :, s] < 0.0)
		dy[:, :, s][bad] *= -1.0

	for s in rear_slots:
		bad = present[:, :, s] & (dy[:, :, s] > 0.0)
		dy[:, :, s][bad] *= -1.0

	# same-lane slots F,R should have small lateral offset
	# clamp |dx| for F and R
	for s in [4, 5]:
		dx[:, :, s] = dx[:, :, s].clip(-5.0, 5.0)

	# adjacent-lane slots should not collapse to absurd dx
	for s in [0, 1, 2, 3]:
		dx[:, :, s] = dx[:, :, s].clip(-20.0, 20.0)

	# clamp dy to plausible range
	dy[:] = dy.clip(-150.0, 150.0)

	X[:, :, 2:20] = nbr.reshape(X.shape[0], X.shape[1], 18)
	return X
