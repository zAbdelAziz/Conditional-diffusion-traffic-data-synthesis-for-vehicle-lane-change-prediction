from typing import Sequence
from os.path import join

from pandas import DataFrame
from numpy import (array, ndarray, arange, zeros, flatnonzero, asarray,
				   concatenate, searchsorted, random, hypot, quantile,
				   int32, int64, float32)

from tqdm import tqdm

from utils.config import Config
from utils.logger import Logger

from datasets.ngsim.interpolate import InterpolateAndSmooth

from numpy import zeros, float32, int32, int64, arange, concatenate, flatnonzero, searchsorted, array, asarray, hypot, quantile, random
from pandas import DataFrame
from typing import Sequence

class DiffusionFeatureBuilder:
	def __init__(self, df: DataFrame):
		self.df = df

		self.name = "ngsim"
		self.cfg = Config().datasets.ngsim.preprocessing
		self.log = Logger(Config().runner.name)

		# Slot indices to place neighbors in a stable order
		self.LF, self.LR, self.RF, self.RR, self.F, self.R = 0, 1, 2, 3, 4, 5

		# frame_index[fid] = {"lanes": {lane_id: lane_pack}, "vmap": {vid: (x,y)}}
		self.frame_index = {}

		# lane_bounds[(Location, ...)] = (min_lane_id, max_lane_id)
		self.lane_bounds = {}

	def build(self):
		interpolator = InterpolateAndSmooth(df=self.df, dt=self.cfg.dt,
											sg_window=self.cfg.smoothing.sg_window, sg_polyorder=self.cfg.smoothing.sg_polyorder)
		self.df = interpolator.run()

		# Save Clean for analysis
		interpolated_path = f'{join(Config().runner.dirs.raw_datasets, f"{self.name}-interpolated-clean")}.csv'
		self.df.to_csv(interpolated_path, index=False)
		self.log.info(f'Saved Interpolated Dataset for analysis in {interpolated_path}')

		# For each frame I built 2 maps
		# lanes[lid] = {vehicle id, x, y}
		# vmap[vehicle] = (x, y)
		# Needed to infer the lanes of the surrounding vehicles [properly]
		self._build_frame_index()

		# Infer Lane Bounds
		self._infer_lane_boundaries(by_cols=("Location",) if "Location" in self.df.columns else ())

		# Build the Sequences
		X, y, meta = self._build_feature_sequences()

		return X, y, meta

	def _build_frame_index(self):
		"""
		Build a per-frame index to enable O(log n) neighbor retrieval
		When creating the features for the surrounding vehicles I need to know [FOR EACH FRAME] the leading and following vehicles to the ego
		so for each frame:
			I loop over each lane and extract 2 dicts:
				lanes[lane_id] stores arrays sorted by Y_s means which vehicles are in this lane from back to forward
					vid[], x[], y[]
				vmap maps Vehicle_ID to (x, y) for quick ID-based lookup
		So when finding the lead/lag vehicles for the ego in each frame I just look up the value of the -1 and +1 vehicles
		"""
		# Group by Frame_ID
		for fid, g in tqdm(self.df.groupby("Frame_ID", sort=False), desc="Frame Indexing"):
			lanes = {}

			# Build per-lane sorted packs (sorted by y for lead/lag lookup)
			for lane_id, gl in g.groupby("Lane_ID", sort=False):
				gl = gl.sort_values("Y_s")
				lanes[lane_id] = {"vid": gl["Vehicle_ID"].to_numpy(int32), "x": gl["X_s"].to_numpy(float32), "y": gl["Y_s"].to_numpy(float32)}

			# Build per-frame vehicle position map for Preceding/Following ID lookups
			vmap = {}
			vids = g["Vehicle_ID"].to_numpy(int32)
			xs = g["X_s"].to_numpy(float32)
			ys = g["Y_s"].to_numpy(float32)

			for i in range(len(vids)):
				vmap[vids[i]] = (xs[i], ys[i])

			self.frame_index[fid] = {"lanes": lanes, "vmap": vmap}

		return self.frame_index

	def _infer_lane_boundaries(self, by_cols: Sequence[str] = ("Location",)):
		"""
		Infer (min_lane_id, max_lane_id) either globally or grouped by metadata
			by_cols: Sequence[str] which are the Columns to group by and If missing I use global bounds
		lane_bounds is a dict: {('Location', ''): (min_lane, max_lane)}
		When computing the features of the neighbors for the ego vehicle I need to make sure that the surrounding lane actually exists
		not to end up ith a lot of nans
		"""
		# If grouping columns are not available, fallback to GLOBAL
		if not by_cols or (not all(c in self.df.columns for c in by_cols)):
			l = self.df["Lane_ID"].dropna().astype(int)
			self.lane_bounds = {("GLOBAL",): (l.min(), l.max())}
			return self.lane_bounds

		out = {}

		# Compute bounds per group key
		for key, g in self.df.groupby(list(by_cols), sort=False):
			key = key if isinstance(key, tuple) else (key,)
			l = g["Lane_ID"].dropna().astype(int)
			if len(l) == 0:
				continue
			out[key] = (l.min(), l.max())

		# Always define a GLOBAL fallback [Just in case it was missing!!]
		l_all = self.df["Lane_ID"].dropna().astype(int)
		out.setdefault(("GLOBAL",), (l_all.min(), l_all.max()))

		self.lane_bounds = out
		return self.lane_bounds

	def _build_feature_sequences(self):
		X_list = []
		y_list = []
		meta_rows = []

		# Generate sequences per vehicle
		for vid, g in tqdm(self.df.groupby("Vehicle_ID", sort=False), desc="Features and Windows"):
			Xv, yv, mv = self._build_vehicle_sequence(processed_vehicle_df=g)
			if Xv is None:
				continue
			X_list.append(Xv)
			y_list.append(yv)
			meta_rows.extend(mv)

		# Concatenate across vehicles
		# [N, T, 14]
		X = concatenate(X_list, axis=0).astype(float32)
		# [N]
		y = concatenate(y_list, axis=0).astype(int64)
		# [N, 3]
		meta = DataFrame(meta_rows, columns=["Vehicle_ID", "End_Frame_ID", "y"])

		return X, y, meta

	def _build_vehicle_sequence(self, processed_vehicle_df: DataFrame):
		"""
        Build windowed feature sequences for a single vehicle [20 features per time step]
        A little bit bloaty but properly documented [Sorry for that]
		"""
		# Cache Config
		seq_len = self.cfg.sequence.length
		horizon = self.cfg.sequence.horizon
		stride = self.cfg.sequence.stride

		# Sort by time
		g = processed_vehicle_df.sort_values("Frame_ID").reset_index(drop=True)
		L = len(g)

		# Need enough frames to create at least one length window and label it using horizon
		if L < seq_len + horizon + 1:
			return None, None, []

		# Define window end indices (inclusive) for sliding windows
		# The Main idea about striding is to create some overlap between samples s that event of the lane change can happen randomly
		# within the horizon not always in the end
		end_indices = arange(seq_len - 1, L - horizon - 1, stride, dtype=int32)
		if end_indices.size == 0:
			return None, None, []

		Nw = end_indices.size
		D = 20

		# Preallocate outputs
		X = zeros((Nw, seq_len, D), dtype=float32)
		y = zeros((Nw,), dtype=int64)
		meta_rows = []

		# Key for per-location lane bounds, if available
		if "Location" in g.columns:
			key = (str(g["Location"].iloc[0]),)
		else:
			key = ("GLOBAL",)

		# Lane bounds for adjacent-lane logic
		min_lane, max_lane = self.lane_bounds.get(key, self.lane_bounds.get(("GLOBAL",), (None, None)))

		# make sure its int
		frames_all = g["Frame_ID"].to_numpy(int32)
		lanes_all = g["Lane_ID"].to_numpy(int32)

		# For Identifying Lane change events I made 2 methods
		# By Horizon and By Boundary [Needs to split the road into segments]
		# Precompute lane-change segments if using boundary-crossing method
		if self.cfg.labeling.method == "horizon":
			segments = None
		else:
			segments = self._find_lane_change_segments(lanes_all, theta_all=g["theta"].to_numpy(float32))

		# Build each window
		for k, end_idx in enumerate(end_indices):
			# Window start index (inclusive)
			s = end_idx - (seq_len - 1)

			# Label this window
			if self.cfg.labeling.method == "horizon":
				y[k] = self._label_by_horizon(lanes_all, end_idx)
			else:
				y[k] = self._label_by_boundary_crossing(end_idx, segments)

			# Slice the observation window
			obs = g.iloc[s: end_idx + 1]

			# Extract arrays for speed
			obs_fid = obs["Frame_ID"].to_numpy(int32)
			obs_lane = obs["Lane_ID"].to_numpy(int32)
			obs_x = obs["X_s"].to_numpy(float32)
			obs_y = obs["Y_s"].to_numpy(float32)
			obs_vx = obs["vx"].to_numpy(float32)
			obs_vy = obs["vy"].to_numpy(float32)

			# Leader / Following Vehicle Ids
			has_pre = "Preceding" in obs.columns
			has_fol = "Following" in obs.columns
			obs_pre = obs["Preceding"].to_numpy(int32) if has_pre else None
			obs_fol = obs["Following"].to_numpy(int32) if has_fol else None

			# Fill per-timestep features
			for t in range(seq_len):
				fid = obs_fid[t]
				lane = obs_lane[t]

				# Ego position ad velocity at this timestep
				ex = obs_x[t]
				ey = obs_y[t]
				evx = obs_vx[t]
				evy = obs_vy[t]

				pack = self.frame_index.get(fid, None)

				# start with all slots absent
				nbr = self._init_empty_neighbors_20()

				if pack is not None:
					lanes_pack = pack["lanes"]
					vmap = pack["vmap"]

					lane_left = lane + 1 if (max_lane is not None and lane < max_lane) else None
					lane_right = lane - 1 if (min_lane is not None and lane > min_lane) else None

					# Left lane neighbors: LF/LR
					if lane_left is not None and lane_left in lanes_pack:
						lead_i, lag_i = self._pick_lane_lead_lag(lanes_pack[lane_left], ey)

						if lead_i >= 0:
							nx = float(lanes_pack[lane_left]["x"][lead_i])
							ny = float(lanes_pack[lane_left]["y"][lead_i])
							nbr[self.LF, :] = (nx - ex, ny - ey, 1.0)

						if lag_i >= 0:
							nx = float(lanes_pack[lane_left]["x"][lag_i])
							ny = float(lanes_pack[lane_left]["y"][lag_i])
							nbr[self.LR, :] = (nx - ex, ny - ey, 1.0)

					# Right lane neighbors: RF/RR
					if lane_right is not None and lane_right in lanes_pack:
						lead_i, lag_i = self._pick_lane_lead_lag(lanes_pack[lane_right], ey)

						if lead_i >= 0:
							nx = float(lanes_pack[lane_right]["x"][lead_i])
							ny = float(lanes_pack[lane_right]["y"][lead_i])
							nbr[self.RF, :] = (nx - ex, ny - ey, 1.0)

						if lag_i >= 0:
							nx = float(lanes_pack[lane_right]["x"][lag_i])
							ny = float(lanes_pack[lane_right]["y"][lag_i])
							nbr[self.RR, :] = (nx - ex, ny - ey, 1.0)

					# Same lane front/rear
					pid = obs_pre[t] if (has_pre and obs_pre is not None) else 0
					rid = obs_fol[t] if (has_fol and obs_fol is not None) else 0

					# Preceding vehicle: front
					if pid != 0 and pid in vmap:
						fx, fy = vmap[pid]
						nbr[self.F, :] = (float(fx) - ex, float(fy) - ey, 1.0)

					# Following vehicle: rear
					if rid != 0 and rid in vmap:
						rx, ry = vmap[rid]
						nbr[self.R, :] = (float(rx) - ex, float(ry) - ey, 1.0)

				# pack 20 dims
				X[k, t, 0:2] = (evx, evy)
				X[k, t, 2:20] = nbr.reshape(-1)

			# Save meta for this window (end frame and label)
			meta_rows.append((g["Vehicle_ID"].iloc[0], frames_all[end_idx], y[k]))

		return X, y, meta_rows

	def _find_lane_change_segments(self, lanes_all: ndarray, theta_all: ndarray):
		"""
		Detect lane-change segments around discrete lane index changes using theta thresholds
		Locate frame indices where Lane_ID changes
		For each change index c:
			Determine lane change direction (delta_sign)
			Walk backward from c until theta exceeds theta_start for n consecutive frames
			Walk forward from c until theta falls below theta_end for n consecutive frames
			Record (start, end, delta_sign) if start < end
		STUPID BUT SAFE
		Output: [(start_idx, end_idx, delta_sign), (...)]
		"""
		L = len(lanes_all)

		# Indices where lane changes (compare consecutive samples)
		changes = flatnonzero(lanes_all[1:] != lanes_all[:-1]) + 1

		segments = []
		for c in changes:
			delta = lanes_all[c] - lanes_all[c - 1]
			delta_sign = 1 if delta > 0 else -1

			# Find segment start by scanning backward
			s = c
			run = 0
			i = c
			while i >= 0:
				if abs(theta_all[i]) >= self.cfg.labeling.theta_start:
					run += 1
					if run >= self.cfg.labeling.consec:
						s = i
				else:
					if run >= self.cfg.labeling.consec:
						break
					run = 0
				i -= 1

			# Find segment end by scanning forward
			e = c
			run = 0
			i = c
			while i < L:
				if abs(theta_all[i]) <= self.cfg.labeling.theta_end:
					run += 1
					if run >= self.cfg.labeling.consec:
						e = i
						break
				else:
					run = 0
				i += 1

			if s < e:
				segments.append((s, e, delta_sign))
		return segments

	def _label_by_boundary_crossing(self, end_idx: int, segments: Sequence):
		"""
		Label based on whether the window end lies inside a detected lane-change segment or not
		"""
		for s, e, delta_sign in segments:
			if s <= end_idx <= e:
				return 2 if delta_sign > 0 else 1
		return 0

	def _label_by_horizon(self, lanes_all: ndarray, end_idx: int):
		"""
		Label based on the first lane change within the future horizon
			lanes_all: Lane_ID series for the vehicle
		SIMPLE AND EASY [but not challenging enough]
		"""
		base_lane = lanes_all[end_idx]
		future = lanes_all[end_idx + 1: end_idx + 1 + self.cfg.sequence.horizon]

		# Find first future index where lane differs
		diff_pos = flatnonzero(future != base_lane)
		if diff_pos.size:
			first_lane = future[diff_pos[0]]
			delta = first_lane - base_lane
			return 2 if delta > 0 else 1
		return 0

	@staticmethod
	def _init_empty_neighbors_20():
		"""
		[6,3] where each row = (dx, dy, p)
		absent => (0, 0, 0)
		"""
		return zeros((6, 3), dtype=float32)

	@staticmethod
	def _pick_lane_lead_lag(lane_pack: dict, ty: float):
		"""
		Given a lane_pack sorted by y, find lead and lag indices around ty
			lane_pack: Contains y-array sorted ascending
			ty: Ego longitudinal position.
		Output:
			(lead_idx, lag_idx)
				lead is the first index with y > ty (ahead)
				lag is the last index with y <= ty (behind)
				If not found index is -1
		"""
		y_arr = lane_pack["y"]

		# First index where y is strictly greater than ty
		j = searchsorted(y_arr, ty, side="right")

		lead = j if j < len(y_arr) else -1
		lag = (j - 1) if (j - 1) >= 0 else -1
		return lead, lag