import json

import numpy as np
import pandas as pd


df = pd.read_csv('ngsim-raw-clean.csv')
cols = ["Local_X", "Local_Y", "Lane_ID", "Preceding", "Following"]


# Splits
splits = {'n_rows':len(df),
		  'n_vehicles': int(df["Vehicle_ID"].nunique()) if "Vehicle_ID" in df else -1,
		  'n_frames': int(df["Frame_ID"].nunique()) if "Frame_ID" in df else -1,
		  'frame_min': int(df["Frame_ID"].min()) if "Frame_ID" in df else -1,
		  'frame_max': int(df["Frame_ID"].max()) if "Frame_ID" in df else -1}
print(splits)


# Missingness
na_rate = {c: float(df[c].isna().mean()) for c in df.columns}

gap_rates = []
max_gaps = []
if "Vehicle_ID" in df.columns and "Frame_ID" in df.columns:
	for _, g in df.groupby("Vehicle_ID", sort=False):
		f = g["Frame_ID"].to_numpy()
		if len(f) < 2:
			continue
		d = np.diff(f)
		gaps = d[d > 1] - 1
		gap_count = int((d > 1).sum())
		gap_rates.append(gap_count / max(1, len(f)))
		max_gaps.append(int(gaps.max()) if len(gaps) else 0)

if len(gap_rates) == 0:
	missing = {'na_rate': na_rate,
			   'gap_rate_per_vehicle_mean': 0.0,
			   'gap_rate_per_vehicle_p95': 0.0,
			   'max_gap_len_p95': 0.0}
else:
	missing = {'na_rate': na_rate,
			   'gap_rate_per_vehicle_mean': float(np.mean(gap_rates)),
			   'gap_rate_per_vehicle_p95': float(np.quantile(gap_rates, 0.95)),
			   'max_gap_len_p95': float(np.quantile(max_gaps, 0.95))}
print(missing)

# Tails
def tail_stats(arr):
	arr = arr[np.isfinite(arr)]
	if arr.size == 0:
		return {'q50': 0., 'q90': 0., 'q99': 0., 'q999': 0., 'mean': 0., 'std': 0.}
	else:
		qs = np.quantile(arr, [0.5, 0.9, 0.99, 0.999])
		return {'q50': qs[0], 'q90': qs[1], 'q99': qs[2], 'q999': qs[3], 'mean': float(arr.mean()), 'std': float(arr.std())}

tails = {}
for c in cols:
	if c in df.columns:
		tails[c] = tail_stats(df[c].to_numpy(dtype=float))
print(tails)

with open('ngsim-raw-csv.stats.txt', 'w') as f:
	stats = {}
	stats['splits'] = splits
	stats['missing'] = missing
	stats['tails'] = tails
	json.dump(stats, f)