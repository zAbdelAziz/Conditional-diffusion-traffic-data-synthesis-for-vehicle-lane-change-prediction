from dataclasses import dataclass

from numpy import ndarray, empty, zeros, zeros_like, concatenate, stack, clip, where, sqrt, arctan2, minimum, float32

from tqdm import tqdm


@dataclass(frozen=True)
class NeighborSlots:
	# left-front
	LF: int = 0
	# left-rear
	LR: int = 1
	# right-front
	RF: int = 2
	# right-rear
	RR: int = 3
	# same-lane front
	F: int = 4
	# same-lane rear
	R: int = 5


class DownstreamFeatureBuilder:
	def __init__(self, dt: float, vision_R: float, tol_absent: float = 1e-3, max_ttc: float = 20.0):
		# Time delta between timesteps
		# frames or seconds must match upstream
		self.dt = float(dt)

		# Radius used for missing-neighbor sentinel and presence thresholding
		self.vision_R = float(vision_R)

		# Small margin so sentinel distance isnt accidentally classified as present
		self.tol_absent = float(tol_absent)

		# Clamp TTC to keep scale bounded and stable
		self.max_ttc = float(max_ttc)

		# Slot index mapping (kept as an object for readability)
		self.slots = NeighborSlots()

		# Small epsilon added to denominators to avoid division-by-zero
		self.eps_denom = 1e-6

	def build(self, X: ndarray, *, batch_size: int = 4096):
		"""Build derived features for an entire dataset"""
		self._validate_input(X)

		N, T, _ = X.shape
		out_dim = 61
		Xd = empty((N, T, out_dim), dtype=float32)

		# Chunk along N for memory-friendly processing
		for s in tqdm(range(0, N, batch_size), desc='Building Downstream Features'):
			e = min(s + batch_size, N)
			Xd[s:e] = self._build_batch(X[s:e])

		return Xd

	def _build_batch(self, Xb: ndarray):
		# Split into ego and neighbors in structured form
		ego_xy, nbr_dxy = self._split_ego_and_neighbors(Xb)

		# Compute neighbor presence mask m ~ {0,1}
		# [N, T, 6]
		m = self._presence_mask(nbr_dxy)

		# Ego kinematics:
		# v, a, jerk, yaw, yaw derivatives
		ego_base, ego_extra = self._ego_features(ego_xy)

		# Neighbor-relative block:
		# dx, dy, dvx, dvy (gated by presence)
		rel = self._relative_features(ego_xy, nbr_dxy, m)

		# Geometry range and bearing per slot (gated)
		geom = self._geometry_features(nbr_dxy, m)

		# Longitudinal TTC per slot (front-closing only)
		ttc = self._ttc_features(rel, m)

		# Gap features for left/right merge decisions
		gap = self._gap_features(rel, ttc, m)

		# Concatenate all
		out = concatenate([ego_base, ego_extra, rel, geom, ttc, gap], axis=-1).astype(float32, copy=False)

		# Sanity: last dim should match the fixed output dimension
		assert out.shape[-1] == 61, f"expected 61 dims, got {out.shape[-1]}"
		return out

	@staticmethod
	def _split_ego_and_neighbors(Xb: ndarray):
		"""
		Split batch into ego positions and neighbor relative offsets
		Output:
			ego_xy: [N, T, 2]
			nbr_dxy: [N, T, 6, 2] (dx, dy per slot)
		"""
		# Ego (x,y) is stored in the first two channels
		ego_xy = Xb[:, :, 0:2].astype(float32)

		# Neighbors occupy channels [2:14]
		nbr_flat = Xb[:, :, 2:14]
		# reshape flat 12 into [6,2]
		nbr_dxy = nbr_flat.reshape(Xb.shape[0], Xb.shape[1], 6, 2).astype(float32)

		return ego_xy, nbr_dxy

	def _presence_mask(self, nbr_dxy: ndarray):
		"""
		Compute presence mask for neighbors based on distance from ego
			Missing neighbors are encoded near radius vision_R
			I marked present if distance < (vision_R - tol_absent)
		Output:
			m: [N, T, 6] float32 with values in {0,1}
		"""
		# r = sqrt(dx^2 + dy^2) per slot
		r = sqrt(nbr_dxy[..., 0] ** 2 + nbr_dxy[..., 1] ** 2).astype(float32)

		# Presence test
		# tol_absent ensures sentinel isnt treated as present
		m = (r < (self.vision_R - self.tol_absent)).astype(float32)
		return m

	def _ego_features(self, ego_xy: ndarray):
		"""
		Compute ego dynamics features based on ego positions x, y
		Output:
			ego_base:  [N, T, 8] = [vx, vy, ax, ay, yaw, yaw_rate, yaw_acc, jerk_x]
			ego_extra: [N, T, 1] = [jerk_y]
		"""
		dt = self.dt

		# Velocity v[t] = (pos[t] - pos[t-1]) / dt
		v = self._finite_difference_1(ego_xy, dt)

		# Acceleration a[t] = (v[t] - v[t-1]) / dt
		a = self._finite_difference_1(v, dt)

		# Jerk j[t] = (a[t] - a[t-1]) / dt
		j = self._finite_difference_1(a, dt)

		# Yaw proxy computed using a 3-frame chord (consistent with upstream).
		yaw = self._yaw_from_chord(ego_xy)

		# Yaw rate and yaw acceleration via finite differences.
		yaw_rate = self._finite_difference_1(yaw[..., None], dt)[..., 0]
		yaw_acc = self._finite_difference_1(yaw_rate[..., None], dt)[..., 0]

		# Unpack ego components (naming for clarity).
		vx, vy = v[..., 0], v[..., 1]
		ax, ay = a[..., 0], a[..., 1]
		jerk_x, jerk_y = j[..., 0], j[..., 1]

		# Compose outputs in the exact expected order.
		ego_base = stack([vx, vy, ax, ay, yaw, yaw_rate, yaw_acc, jerk_x], axis=-1).astype(float32)

		ego_extra = jerk_y[..., None].astype(float32)

		return ego_base, ego_extra

	@staticmethod
	def _yaw_from_chord(ego_xy: ndarray) -> ndarray:
		"""
		Compute yaw-like proxy angle using a 3-frame chord
			Uses atan2(dx, dy) (angle w.r.t +Y axis)
		Output:
			yaw: [N, T]
		"""
		N, T, _ = ego_xy.shape
		yaw = zeros((N, T), dtype=float32)
		# Need at least 4 timesteps to compute a 3-step chord at index 3
		if T > 3:
			dx = ego_xy[:, 3:, 0] - ego_xy[:, :-3, 0]
			dy = ego_xy[:, 3:, 1] - ego_xy[:, :-3, 1]
			yaw[:, 3:] = arctan2(dx, dy).astype(float32)
		return yaw

	def _relative_features(self, ego_xy: ndarray, nbr_dxy: ndarray, m: ndarray):
		"""
		Build relative interaction features per slot:
			[dx, dy, dvx, dvy] × 6 slots = 24 dims
			dvx/dvy are computed from time derivative of relative coords (nbr_dxy) and then compared against ego velocity
			all gated by presence
		Output:
			rel: [N, T, 24]
		"""
		dt = self.dt

		# Ego velocity in global coordinates (derived from ego_xy)
		# [N,T,2]
		ego_v = self._finite_difference_1(ego_xy, dt)

		# Neighbor "velocity" on relative coords:
		# Only valid when neighbor is present at both t-1 and t.
		# [N,T,6,2]
		nbr_v = zeros_like(nbr_dxy, dtype=float32)
		if nbr_dxy.shape[1] > 1:
			# valid[t] = m[t]*m[t-1]
			# expanded for dx/dy channels
			# [N,T-1,6,1]
			valid = (m[:, 1:, :] * m[:, :-1, :]).astype(float32)[:, :, :, None]

			# Relative-Coordinate Difference
			# dv = (nbr[t] - nbr[t-1]) / dt
			dv = (nbr_dxy[:, 1:, :, :] - nbr_dxy[:, :-1, :, :]) / dt  # [N,T-1,6,2]

			# Gate by valid to suppress spikes when slot appears/disappears
			nbr_v[:, 1:, :, :] = dv * valid

		# Gated relative position
		# [N,T,6]
		dx_rel = nbr_dxy[..., 0] * m
		# [N,T,6]
		dy_rel = nbr_dxy[..., 1] * m

		# Relative velocity difference vs ego velocity (broadcast ego_v over slots)
		# [N,T,6]
		dvx = (nbr_v[..., 0] - ego_v[..., 0][:, :, None]) * m
		dvy = (nbr_v[..., 1] - ego_v[..., 1][:, :, None]) * m

		# Pack per-slot features in fixed order and flatten slots
		# [N,T,6,4]
		rel = stack([dx_rel, dy_rel, dvx, dvy], axis=-1).astype(float32)
		# [N,T,24]
		rel = rel.reshape(rel.shape[0], rel.shape[1], 24)

		return rel

	@staticmethod
	def _geometry_features(nbr_dxy: ndarray, m: ndarray) -> ndarray:
		"""
		Compute geometry features per slot:
			range = sqrt(dx^2 + dy^2)
			bearing = atan2(dx, dy)  to be consistent with yaw convention [+Y]
			Both are gated by presence.
		Output:
			geom: [N,T,12] = [r(6), bearing(6)]
		"""
		# Apply presence gating on dx/dy before computing geometry
		dx = nbr_dxy[..., 0] * m
		dy = nbr_dxy[..., 1] * m

		# Range
		# missing neighbors become 0 because dx/dy were gated
		r = sqrt(dx * dx + dy * dy).astype(float32)  # [N,T,6]

		# Bearing
		# multiply by m to ensure missing = 0
		bearing = arctan2(dx, dy).astype(float32) * m  # [N,T,6]

		# Concatenate per-slot ranges and bearings
		# [N,T,12]
		geom = concatenate([r, bearing], axis=-1).astype(float32)
		return geom

	def _ttc_features(self, rel: ndarray, m: ndarray) -> ndarray:
		"""
		Compute simple longitudinal Time To Collision per slot
		Convention:
			Use dy and dvy (ego frame)
			TTC applies only if dy>0 (neighbor in front) and dvy<0 (closing)
			TTC = dy / (-dvy + eps)
			Clamp to [0, max_ttc]
			Missing or invalid => 0
		Outputs:
			ttc: [N,T,6]
		"""
		# rel is [dx, dy, dvx, dvy] repeated per slot
		# [N,T,6,4]
		rel4 = rel.reshape(rel.shape[0], rel.shape[1], 6, 4)
		# Unpack dy and dvy per slot from rel block
		# [N,T,6]
		dy = rel4[..., 1]
		# [N,T,6]
		dvy = rel4[..., 3]

		ttc = zeros_like(dy, dtype=float32)

		# Valid TTC when neighbor is present in front and closing
		valid = (m > 0.0) & (dy > 0.0) & (dvy < 0.0)

		# Avoid division by zero
		# denom is positive when dvy<0
		denom = (-dvy) + self.eps_denom
		ttc_raw = dy / denom

		# Assign valid TTCs
		ttc[valid] = ttc_raw[valid].astype(float32)

		# Clamp for stability
		if self.max_ttc is not None and self.max_ttc > 0:
			ttc = clip(ttc, 0.0, self.max_ttc, out=ttc)

		return ttc

	def _gap_features(self, rel: ndarray, ttc_front: ndarray, m: ndarray) -> ndarray:
		"""
		Compute 10 merge-relevant gap features:
			Left side  (LF, LR):  [front_gap, rear_gap, closing_front, closing_rear, min_ttc]
			Right side (RF, RR):  [front_gap, rear_gap, closing_front, closing_rear, min_ttc]
		Definitions:
			front_gap: dy_front if present and dy>0 else 0
			rear_gap:  -dy_rear if present and dy<0 else 0
			closing_front: -dvy_front if present else 0
				positive means closing
			closing_rear:   dvy_rear if present else 0
				rear approaching means positive
		  	min_ttc: min(front TTC, rear TTC)
		  		with a rear-TTC computed separately
		Returns:
			gap_feat: [N,T,10]
		"""
		s = self.slots

		# [N,T,6,4]
		rel4 = rel.reshape(rel.shape[0], rel.shape[1], 6, 4)
		# [N,T,6]
		dy = rel4[..., 1]
		# [N,T,6]
		dvy = rel4[..., 3]

		# Left side (LF, LR)
		dy_LF, dy_LR = dy[:, :, s.LF], dy[:, :, s.LR]
		dvy_LF, dvy_LR = dvy[:, :, s.LF], dvy[:, :, s.LR]
		m_LF, m_LR = m[:, :, s.LF], m[:, :, s.LR]

		front_gap_L = self.front_gap(dy_LF, m_LF)
		rear_gap_L = self.rear_gap(dy_LR, m_LR)
		closing_front_L = self.closing_front_rate(dvy_LF, m_LF)
		closing_rear_L = self.closing_rear_rate(dvy_LR, m_LR)

		ttc_front_L = ttc_front[:, :, s.LF]            # front TTC already computed
		ttc_rear_L = self.rear_ttc(dy_LR, dvy_LR, m_LR)     # rear TTC computed here

		# min_ttc_L: minimum of valid TTCs
		# if neither valid then I set to 0
		min_ttc_L = minimum(where(ttc_front_L > 0.0, ttc_front_L, float32(self.max_ttc)), where(ttc_rear_L > 0.0, ttc_rear_L, float32(self.max_ttc))).astype(float32)
		neither_L = (ttc_front_L <= 0.0) & (ttc_rear_L <= 0.0)
		min_ttc_L[neither_L] = 0.0

		# Right side (RF, RR)
		dy_RF, dy_RR = dy[:, :, s.RF], dy[:, :, s.RR]
		dvy_RF, dvy_RR = dvy[:, :, s.RF], dvy[:, :, s.RR]
		m_RF, m_RR = m[:, :, s.RF], m[:, :, s.RR]

		front_gap_R = self.front_gap(dy_RF, m_RF)
		rear_gap_R = self.rear_gap(dy_RR, m_RR)
		closing_front_R = self.closing_front_rate(dvy_RF, m_RF)
		closing_rear_R = self.closing_rear_rate(dvy_RR, m_RR)

		ttc_front_R = ttc_front[:, :, s.RF]
		ttc_rear_R = self.rear_ttc(dy_RR, dvy_RR, m_RR)

		min_ttc_R = minimum(where(ttc_front_R > 0.0, ttc_front_R, float32(self.max_ttc)), where(ttc_rear_R > 0.0, ttc_rear_R, float32(self.max_ttc)),).astype(float32)

		neither_R = (ttc_front_R <= 0.0) & (ttc_rear_R <= 0.0)
		min_ttc_R[neither_R] = 0.0

		# Pack gap features in stable order
		# left(5) then right(5)
		gap_feat = stack([front_gap_L, rear_gap_L, closing_front_L, closing_rear_L, min_ttc_L,
						  front_gap_R, rear_gap_R, closing_front_R, closing_rear_R, min_ttc_R,], axis=-1).astype(float32)  # [N,T,10]
		return gap_feat

	@staticmethod
	def _finite_difference_1(x: ndarray, dt: float):
		"""First-order finite difference along time axis with zero at t=0"""
		dxdt = zeros_like(x, dtype=float32)
		# Forward difference for t>=1 and t=0 stays 0
		if x.shape[1] > 1:
			dxdt[:, 1:, ...] = (x[:, 1:, ...] - x[:, :-1, ...]) / dt
		return dxdt

	@staticmethod
	def front_gap(dy_slot: ndarray, m_slot: ndarray) -> ndarray:
		out = zeros_like(dy_slot, dtype=float32)
		ok = (m_slot > 0.0) & (dy_slot > 0.0)
		out[ok] = dy_slot[ok].astype(float32)
		return out

	@staticmethod
	def rear_gap(dy_slot: ndarray, m_slot: ndarray) -> ndarray:
		out = zeros_like(dy_slot, dtype=float32)
		ok = (m_slot > 0.0) & (dy_slot < 0.0)
		out[ok] = (-dy_slot[ok]).astype(float32)
		return out

	@staticmethod
	def closing_front_rate(dvy_slot: ndarray, m_slot: ndarray) -> ndarray:
		out = zeros_like(dvy_slot, dtype=float32)
		ok = (m_slot > 0.0)
		out[ok] = (-dvy_slot[ok]).astype(float32)  # positive => closing
		return out

	@staticmethod
	def closing_rear_rate(dvy_slot: ndarray, m_slot: ndarray) -> ndarray:
		out = zeros_like(dvy_slot, dtype=float32)
		ok = (m_slot > 0.0)
		out[ok] = (dvy_slot[ok]).astype(float32)
		return out

	def rear_ttc(self, dy_slot: ndarray, dvy_slot: ndarray, m_slot: ndarray) -> ndarray:
		"""
		TTC for a rear vehicle (behind ego) that is closing:
		  dy < 0 and dvy > 0  => TTC = (-dy) / (dvy + eps)
		Else 0.
		"""
		out = zeros_like(dy_slot, dtype=float32)
		ok = (m_slot > 0.0) & (dy_slot < 0.0) & (dvy_slot > 0.0)
		denom = (dvy_slot + self.eps_denom)
		raw = (-dy_slot) / denom
		out[ok] = raw[ok].astype(float32)
		if self.max_ttc is not None and self.max_ttc > 0:
			out = clip(out, 0.0, self.max_ttc, out=out)
		return out

	@staticmethod
	def _validate_input(X: ndarray):
		assert isinstance(X, ndarray), "X must be a numpy array"
		assert X.ndim == 3, f"expected [N,T,D], got {X.shape}"
		assert X.shape[-1] == 14, f"expected last dim 14, got {X.shape[-1]}"