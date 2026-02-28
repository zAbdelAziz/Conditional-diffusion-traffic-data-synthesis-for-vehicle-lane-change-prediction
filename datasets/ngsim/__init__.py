from pandas import DataFrame

from datasets.base import BaseDataset


class NgsimDataset(BaseDataset):
	def __init__(self, name: str, raw: bool = False):
		super().__init__(name=name, raw=raw)

		# # TODO Enable after cleaning the _build_refactored
		# # X Shape: [n_samples, n_lookback, n_features]
		# # Numer of Timesteps
		# self.T = self.X.shape[1]
		# # Number of features
		# self.D = self.X.shape[2]

	def _build_refactored(self):
		self._read_raw()

	def _clean_raw_csv(self, df: DataFrame):
		return df