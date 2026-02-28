from numpy import rint
from pandas import DataFrame

from datasets.base import BaseDataset
from datasets.ngsim.diffusion_feature_builder import DiffusionFeatureBuilder
from utils.misc import is_all
from utils.pd_utils import de_to_numeric


class NgsimDataset(BaseDataset):
	def __init__(self, name: str, raw: bool = False):
		super().__init__(name=name, raw=raw)

		# X Shape: [n_samples, n_lookback, n_features]
		# Numer of Timesteps
		self.T = self.X.shape[1]
		# Number of features
		self.D = self.X.shape[2]

	def _build_refactored(self):
		# Read the Raw Dataset
		df = self._read_raw()

		# I Split Refactoring in another class [cuz its big]
		# Converts the csv to a time series
		feature_builder = DiffusionFeatureBuilder(df = df)
		self.X, self.y, self.meta = feature_builder.build()

		# Balance Classes
		if self.cfg.datasets[self.name].preprocessing.balance_classes:
			self.balance_classes()

		# Save
		self.save()

		self.vision_R = feature_builder.vision_R

	def _clean_raw_csv(self, df: DataFrame):
		# Select Subsets based on config
		subsets = getattr(self.cfg.datasets[self.name], "subsets", None)
		if subsets is not None:
			df = self._select_subsets(df, subsets)

		# Convert All Numeric Columns to Standard format instead of DE Numeric Format
		df = de_to_numeric(df=df, columns=df.columns.tolist())

		# Drop NA
		df = df.dropna().copy()

		# Cast Vehicle_ID, Lane_ID, Frame_ID, Preceding, Following to Integer columns
		df = self._cast_int(df=df)

		# Sort Values by Vehicle_ID and Frame_ID
		df = df.sort_values(["Vehicle_ID", "Frame_ID"]).reset_index(drop=True)

		self.log.info(f"NGSIM loaded rows after subsets: {len(df)} | vehicles: {df['Vehicle_ID'].nunique()}")

		return df

	def _select_subsets(self, df: DataFrame, subsets: dict):
		# Location subsets (string compare and case-insensitive)
		self.log.info(f'Selecting subsets: {subsets}')
		if "Location" in df.columns and (not is_all(getattr(subsets, "locations", "*"))):
			allowed = getattr(subsets, "locations")
			allowed = [str(a).strip().lower() for a in allowed]
			df = df[df["Location"].astype(str).str.strip().str.lower().isin(allowed)]
		return df

	def _cast_int(self, df: DataFrame):
		if 'Vehicle_ID' in df.columns:
			self.log.info(f'Casting Vehicle_ID to int')
			df["Vehicle_ID"] = df["Vehicle_ID"].astype(int)
		if 'Frame_ID' in df.columns:
			self.log.info(f'Casting Frame_ID to int')
			df["Frame_ID"] = df["Frame_ID"].astype(int)
		if 'Lane_ID' in df.columns:
			self.log.info(f'Casting Lane_ID to int')
			df["Lane_ID"] = df["Lane_ID"].astype(int)
		# Just to be Sure
		if "Preceding" in df.columns:
			self.log.info(f'Casting Preceding to int')
			df["Preceding"] = rint(df["Preceding"].fillna(0.0)).astype(int)
		if "Following" in df.columns:
			self.log.info(f'Casting Following to int')
			df["Following"] = rint(df["Following"].fillna(0.0)).astype(int)
		return df
