from numpy import ndarray, float32, zeros, where, nan_to_num


def postprocess_synth_20(
	X: ndarray,
	*,
	p_threshold: float = 0.5,
	smooth_p: bool = False,
	hysteresis: bool = True,
	p_on: float = 0.55,
	p_off: float = 0.45,
	same_lane_dx_max: float = 79.497,
	adjacent_lane_dx_max: float = 93.747,
	dy_abs_max: float = 748.17,
	vx_min: float = -25.075,
	vx_max: float = 25.937,
	vy_min: float = -407.907,
	vy_max: float = 224.917,
):
	"""
	Conservative projection of generated [N,T,20] traffic sequences.

	Expected to run on PHYSICAL units, i.e. after inverse standardization.

	layout:
	  [vx, vy,
	   LF_dx, LF_dy, LF_p,
	   LR_dx, LR_dy, LR_p,
	   RF_dx, RF_dy, RF_p,
	   RR_dx, RR_dy, RR_p,
	   F_dx,  F_dy,  F_p,
	   R_dx,  R_dy,  R_p]
	"""
	if X.ndim != 3 or X.shape[-1] != 20:
		raise ValueError(f"expected [N,T,20], got {X.shape}")

	# sanitize
	X = nan_to_num(X.astype(float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)

	# ego clamps using plausible bounds from dataset
	if vx_min is not None or vx_max is not None:
		lo = -1e9 if vx_min is None else float(vx_min)
		hi = +1e9 if vx_max is None else float(vx_max)
		X[:, :, 0] = X[:, :, 0].clip(lo, hi)

	if vy_min is not None or vy_max is not None:
		lo = -1e9 if vy_min is None else float(vy_min)
		hi = +1e9 if vy_max is None else float(vy_max)
		X[:, :, 1] = X[:, :, 1].clip(lo, hi)

	nbr = X[:, :, 2:20].reshape(X.shape[0], X.shape[1], 6, 3)
	dx = nbr[..., 0]
	dy = nbr[..., 1]
	p = nbr[..., 2]

	# keep mask probabilities valid
	p[:] = p.clip(0.0, 1.0)

	# small temporal smoothing on soft p before binarization
	if smooth_p and X.shape[1] >= 3:
		p2 = p.copy()
		p2[:, 1:-1, :] = (p[:, :-2, :] + p[:, 1:-1, :] + p[:, 2:, :]) / 3.0
		p2[:, 0, :] = (2.0 * p[:, 0, :] + p[:, 1, :]) / 3.0
		p2[:, -1, :] = (p[:, -2, :] + 2.0 * p[:, -1, :]) / 3.0
		p[:] = p2.clip(0.0, 1.0)

	# presence decision
	if hysteresis and X.shape[1] >= 2:
		if not (0.0 <= p_off <= p_on <= 1.0):
			raise ValueError(f"expected 0 <= p_off <= p_on <= 1, got p_off={p_off}, p_on={p_on}")

		present = zeros(p.shape, dtype=bool)

		# initialize from standard threshold
		state = p[:, 0, :] > p_threshold
		present[:, 0, :] = state

		# hysteresis over time: high prob turns on, low prob turns off
		# ambiguous region keeps previous state
		for t in range(1, X.shape[1]):
			turn_on = p[:, t, :] >= p_on
			turn_off = p[:, t, :] <= p_off
			state = where(turn_on, True, where(turn_off, False, state))
			present[:, t, :] = state
	else:
		present = p > p_threshold

	# slot semantics:
	# front slots: LF, RF, F then dy >= 0
	# rear  slots: LR, RR, R then dy <= 0
	front_slots = [0, 2, 4]
	rear_slots = [1, 3, 5]

	# apply sign convention
	# done before the final absent-zeroing, so absent slots will still end up exactly zero
	dy[:, :, front_slots] = abs(dy[:, :, front_slots])
	dy[:, :, rear_slots] = -abs(dy[:, :, rear_slots])

	# same-lane slots F/R should have small lateral offset
	if same_lane_dx_max is not None:
		m = float(same_lane_dx_max)
		dx[:, :, 4:6] = dx[:, :, 4:6].clip(-m, m)

	# adjacent-lane slots should not explode laterally
	if adjacent_lane_dx_max is not None:
		m = float(adjacent_lane_dx_max)
		dx[:, :, 0:4] = dx[:, :, 0:4].clip(-m, m)

	# longitudinal range clamp
	if dy_abs_max is not None:
		m = float(dy_abs_max)
		dy[:] = dy.clip(-m, m)

	# final hard projection:
	# absent means zero geometry binary mask
	dx[~present] = 0.0
	dy[~present] = 0.0
	p[:] = present.astype(float32)

	X[:, :, 2:20] = nbr.reshape(X.shape[0], X.shape[1], 18)
	return X