import json

import numpy as np


def refactored_stats(X, y, vision_R, slot_signs):
	if X is None or y is None:
		return {}

	N, T, D = X.shape
	out = {"shape": {"N": int(N), "T": int(T), "D": int(D)}}

	flat = X.reshape(-1, D)
	out["per_dim_std"] = flat.std(axis=0).tolist()
	out["per_dim_q99_abs"] = np.quantile(np.abs(flat), 0.99, axis=0).tolist()

	if vision_R is not None:
		R = float(vision_R)
		# neighbor dy dims in your encoding:
		# features 2:14 => 12 dims, arranged as 6 rows (dx,dy)
		neigh = X[..., 2:14].reshape(N, T, 6, 2)
		dy = neigh[..., 1]  # [N,T,6]
		# missing if dy == slot_sign * R AND dx == 0
		dx = neigh[..., 0]
		if slot_signs is not None:
			target = slot_signs.reshape(1, 1, 6) * R
			miss = (dx == 0.0) & (np.isclose(dy, target, atol=1e-5))
		else:
			# fallback: treat abs(dy) close to R as missing
			miss = (dx == 0.0) & (np.isclose(np.abs(dy), R, atol=1e-5))

		miss_rate_slot = miss.mean(axis=(0 ,1))  # [6]
		out["neighbor_missing_rate_slot"] = miss_rate_slot.tolist()

		# by class
		out["neighbor_missing_rate_by_class"] = {}
		for cls in np.unique(y):
			m = miss[y == cls]
			out["neighbor_missing_rate_by_class"][int(cls)] = m.mean(axis=(0 ,1)).tolist()

	return out


ds = np.load('ngsim.npz')
X = ds["X"]
y = ds["y"]

vision_R = np.load('ngsim.std.npz')['vision_R']
slot_signs = np.array([+1, -1, +1, -1, +1, -1])

stats = refactored_stats(X, y, vision_R, slot_signs)
print(stats)

with open("ngsim.stats.json", 'w') as f:
	json.dump(stats, f)
