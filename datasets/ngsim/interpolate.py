from numpy import zeros, ndarray, arctan2, arange, int32, float32, float64
from pandas import DataFrame, to_numeric, concat

from scipy.signal import savgol_filter

from tqdm import tqdm

class InterpolateAndSmooth:
	def __init__(self, df: DataFrame, dt: int, sg_window: int, sg_polyorder: int):
		self.df = df

		self.dt = dt
		self.sg_window = sg_window
		self.sg_polyorder = sg_polyorder

	def run(self):
		"""
		For each vehicle:
			Make frames contiguous (reindex Frame_ID)
			Interpolate Local_X/Local_Y linearly for missing frames
			Apply Savitzky-Golay smoothing (if enough samples)
			Re-Compute velocities/accelerations and yaw proxy
		"""

		# list of dataframes
		vehicles = []

		# Group by vehicle and process each track independently
		for vid, g in tqdm(self.df.groupby("Vehicle_ID", sort=False), desc="Interpolation & Smoothing"):
			g2 = self._interpolate_and_smooth_vehicle(g)

			# Preserve constant metadata columns if present in original but not in output
			for col in ["Location", "Section_ID", "Direction"]:
				if col in g.columns and col not in g2.columns:
					g2[col] = g[col].iloc[0]

			vehicles.append(g2)

		# Replace the builder's dataframe with the processed version
		self.df = concat(vehicles, axis=0, ignore_index=True)
		return self.df

	def _interpolate_and_smooth_vehicle(self, g: DataFrame):
		"""
		Process one vehicle trajectory:
			Ensure unique Frame_ID rows [Drop duplicates just in case]
			Reindex to full contiguous frame range [make sure theres no missing frames / rows]
			Fill categorical/id columns [instead of dropping NAs, just a backward/forward filling for categorical "Lane_ID, Preceding, Following"]
			Interpolate positions [Linearly]
			Smooth positions [using savgol filter just to be consistent with the sota, but I think other smoothing functions are also valid]
			Compute kinematics [Velocity/Accelaration] and yaw
		"""
		# Sort by time and keep one row per Frame_ID
		g = g.sort_values("Frame_ID").reset_index(drop=True)
		g = g.drop_duplicates(subset=["Frame_ID"], keep="last").reset_index(drop=True)

		# Build full contiguous frame range for this vehicle
		fmin, fmax = int(g["Frame_ID"].iloc[0]), int(g["Frame_ID"].iloc[-1])
		full_frames = arange(fmin, fmax + 1, dtype=int32)

		# Reindex: missing frames become NaN rows
		g_full = g.set_index("Frame_ID").reindex(full_frames)
		g_full.index.name = "Frame_ID"
		g_full = g_full.reset_index()

		# Vehicle_ID is constant so I set it explicitly for reindexed rows
		g_full["Vehicle_ID"] = int(g["Vehicle_ID"].iloc[0])

		# Lane_ID is treated as piecewise-constant so fill gaps forward/backward
		if "Lane_ID" in g_full.columns:
			g_full["Lane_ID"] = g_full["Lane_ID"].ffill().bfill()

		# Preceding/Following IDs are also treated as piecewise-constant
		for col in ["Preceding", "Following"]:
			if col in g_full.columns:
				g_full[col] = g_full[col].ffill().bfill()

		# Ensure positions are numeric and interpolate linearly across gaps
		for col in ["Local_X", "Local_Y"]:
			g_full[col] = to_numeric(g_full[col], errors="coerce")
			g_full[col] = g_full[col].interpolate(method="linear", limit_direction="both")

		# Choose a valid Savitzky-Golay window (odd and <= length)
		L = len(g_full)
		win = min(self.sg_window, self._largest_odd_leq(L))

		# If too short, skip SG smoothing and keep interpolated signals
		if win < 5:
			x_s = g_full["Local_X"].to_numpy(float32)
			y_s = g_full["Local_Y"].to_numpy(float32)
		else:
			# Use float64 for filtering stability then cast to float32
			x_raw = g_full["Local_X"].to_numpy(float64)
			y_raw = g_full["Local_Y"].to_numpy(float64)

			# polyorder must be < window_length
			poly = min(self.sg_polyorder, win - 2)

			x_s = savgol_filter(x_raw, window_length=win, polyorder=poly, mode="interp").astype(float32)
			y_s = savgol_filter(y_raw, window_length=win, polyorder=poly, mode="interp").astype(float32)

		# Store smoothed positions
		g_full["X_s"] = x_s
		g_full["Y_s"] = y_s

		# Derive kinematics and yaw proxy from smoothed positions
		g_full = self._calculate_velocity_acceleration(g=g_full, x_s=x_s, y_s=y_s)
		g_full = self._calculate_yaw(g=g_full, x_s=x_s, y_s=y_s)

		# Type normalization
		g_full["Frame_ID"] = g_full["Frame_ID"].astype(int32)
		if "Lane_ID" in g_full.columns:
			g_full["Lane_ID"] = to_numeric(g_full["Lane_ID"], errors="coerce").astype("Int64")

		return g_full

	def _calculate_velocity_acceleration(self, g: DataFrame, x_s: ndarray, y_s: ndarray):
		"""
		Compute first-order (v) and second-order (a) finite differences
			The first element is set to 0 since I don't have t-1 for t=0
			Assumes uniform dt
			Found this method in a paper from the references
		I dont rely on the velocity of the dataset because its too noisy and almost unreliable [same as the sota]
		"""
		L = len(g)

		# Preallocate arrays
		vx = zeros(L, dtype=float32)
		vy = zeros(L, dtype=float32)
		ax = zeros(L, dtype=float32)
		ay = zeros(L, dtype=float32)

		# Velocity: forward difference (starting at index 1)
		vx[1:] = (x_s[1:] - x_s[:-1]) / self.dt
		vy[1:] = (y_s[1:] - y_s[:-1]) / self.dt

		# Acceleration: difference of velocity
		ax[1:] = (vx[1:] - vx[:-1]) / self.dt
		ay[1:] = (vy[1:] - vy[:-1]) / self.dt

		# Attach to dataframe
		g["vx"] = vx
		g["vy"] = vy
		g["ax"] = ax
		g["ay"] = ay
		return g

	@staticmethod
	def _calculate_yaw(g: DataFrame, x_s: ndarray, y_s: ndarray):
		"""
		Compute a yaw-like proxy angle theta
		Implementation
			I Used a 3-frame chord:
				dx = x[t] - x[t-3]
				dy = y[t] - y[t-3]
				theta[t] = atan2(dx, dy)
		I use atan2(dx, dy) instead of atan2(dy, dx) to calculate the angle measured from the +Y-axis not the +X-axis
		TODO Make the 3-frame a hyper parameter?
		"""
		L = len(g)
		theta = zeros(L, dtype=float32)

		# Need at least 4 samples to form a 3-step difference at index 3
		if L > 3:
			dx = x_s[3:] - x_s[:-3]
			dy = y_s[3:] - y_s[:-3]
			theta[3:] = arctan2(dx, dy).astype(float32)

		g["theta"] = theta
		return g

	@staticmethod
	def _largest_odd_leq(n: int):
		# Largest Odd Integer [used in Savitzky-Golay window length]
		return n if (n % 2 == 1) else (n - 1)