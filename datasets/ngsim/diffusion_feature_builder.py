from typing import Sequence
from numpy import int32, float32
from pandas import DataFrame

from tqdm import tqdm

from datasets.ngsim.interpolate import InterpolateAndSmooth
from utils.config import Config


class DiffusionFeatureBuilder:
	def __init__(self, df: DataFrame):
		self.df = df

		self.name = "ngsim"
		self.cfg = Config().datasets.ngsim.preprocessing

		# Slot indices to place neighbors in a stable order
		self.LF, self.LR, self.RF, self.RR, self.F, self.R = 0, 1, 2, 3, 4, 5

		# frame_index[fid] = {"lanes": {lane_id: lane_pack}, "vmap": {vid: (x,y)}}
		self.frame_index = {}

		# lane_bounds[(Location, ...)] = (min_lane_id, max_lane_id)
		self.lane_bounds = {}

	def build(self):
		# Interpolates the Local_X, Local_Y linearly for missing frames
		# Apply Savitzky-Golay smoothing
		# Compute velocities/accelerations and yaw proxy
		interpolator = InterpolateAndSmooth(df=self.df, dt=self.cfg.dt, sg_window=self.cfg.sg_window, sg_polyorder=self.cfg.sg_polyorder)
		self.df = interpolator.run()

		# For each frame I built 2 maps
		# lanes[lid] = {vehicle id, x, y}
		# vmap[vehicle] = (x, y)
		# Needed to infer the lanes of the surrounding vehicles [properly]
		self.build_frame_index()

		# Infer Lane Bounds
		self.infer_lane_boundaries(by_cols=("Location",) if "Location" in self.df.columns else ())


	def build_frame_index(self):
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
				lanes[int(lane_id)] = {"vid": gl["Vehicle_ID"].to_numpy(int32), "x": gl["X_s"].to_numpy(float32), "y": gl["Y_s"].to_numpy(float32)}

			# Build per-frame vehicle position map for Preceding/Following ID lookups
			vmap = {}
			vids = g["Vehicle_ID"].to_numpy(int32)
			xs = g["X_s"].to_numpy(float32)
			ys = g["Y_s"].to_numpy(float32)

			for i in range(len(vids)):
				vmap[int(vids[i])] = (float(xs[i]), float(ys[i]))

			self.frame_index[int(fid)] = {"lanes": lanes, "vmap": vmap}

		return self.frame_index

	def infer_lane_boundaries(self, by_cols: Sequence[str] = ("Location",)):
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
			self.lane_bounds = {("GLOBAL",): (int(l.min()), int(l.max()))}
			return self.lane_bounds

		out = {}

		# Compute bounds per group key
		for key, g in self.df.groupby(list(by_cols), sort=False):
			key = key if isinstance(key, tuple) else (key,)
			l = g["Lane_ID"].dropna().astype(int)
			if len(l) == 0:
				continue
			out[key] = (int(l.min()), int(l.max()))

		# Always define a GLOBAL fallback [Just in case it was missing!!]
		l_all = self.df["Lane_ID"].dropna().astype(int)
		out.setdefault(("GLOBAL",), (int(l_all.min()), int(l_all.max())))

		self.lane_bounds = out
		return self.lane_bounds
