from os.path import join
from pathlib import Path

from numpy import savez_compressed, load, random, flatnonzero, concatenate, bincount, int64
from pandas import read_csv, DataFrame
from torch.utils.data import Dataset

from utils.config import Config
from utils.logger import Logger


class BaseDataset(Dataset):
	def __init__(self, name: str, raw: bool = False):
		self.name = name
		self.raw = raw
		
		self.cfg = Config()
		self.log = Logger(self.cfg.runner.name)
		
		# Build Paths from config [Raw, Refactored, Synthetic, Standardizers]	
		self.paths = self._resolve_paths()
		
		# Build Refactored [Timeseries] dataset from raw csv
		# Logic Only [actual refactoring is done by the child class]
		if self.raw:
			# If Refactored Dataset is not available we build it
			if (not self.paths['ref_meta'].exists()) or (not self.paths['ref_npz'].exists()):
				self._build_refactored()
			# Otherwise do nothing [Loads the dataset from child class]
		else:
			# If Synthetic Data is not available we fall back to refactored raw data [Flip the raw to True]
			if (not self.paths['syn_meta'].exists()) or (not self.paths['syn_npz'].exists()):
				self.log.error(f'Synthetic data for {self.name} dataset doesnt exist so falling back to raw data')
				self.raw = True
				# If Refactored Dataset is not available we build it
				if (not self.paths['ref_meta'].exists()) or (not self.paths['ref_npz'].exists()):
					self._build_refactored()
				# Otherwise do nothing [Loads the dataset from child class]
			# Otherwise [meta, npz] exists so we keep the raw as false [Loads the dataset from child class]

		# Vision Threshold
		self.vision_R = None

		# Load the Refactored/Synthetic Dataset
		# # TODO Enable after cleaning the _build_refactored
		# self.X, self.y, self.meta = self._load_refactored()

	def _build_refactored(self):
		# Different Datasets Has different Preprocessing and cleaning
		raise NotImplementedError('Should be implemented by child dataset')

	def _load_refactored(self):
		# Load the refactored/synthetic datasets from disk
		# Should be the same logic in all datasets
		if self.raw:
			# Load refactored
			meta = read_csv(self.paths['ref_meta'])
			data = load(self.paths['ref_npz'], allow_pickle=False)
		else:
			# Load Synth
			meta = read_csv(self.paths['syn_meta'])
			data = load(self.paths['syn_npz'], allow_pickle=False)
		
		# Sanity checks
		if len(meta) != data['X'].shape[0] or data['X'].shape[0] != data['y'].shape[0]:
			raise RuntimeError(f'Cache mismatch: meta={len(meta)}, X={data['X'].shape}, y={data['y'].shape}')
		
		return data['X'], data['y'], meta

	def _read_raw(self):
		# Read RAW CSV and clean it
		# Logic Should be identical in all datasets [Cleaning is not]
		raw_path = self.paths["raw"]
		if not raw_path.exists():
			raise FileNotFoundError(f"dataset {self.name} raw file not found: {raw_path}")

		# Load selected columns as strings [the values/number of ngsim are in eu format! not sure why! so I just read as str and made a clean method]
		self.log.info('Reading Raw CSV')

		columns = getattr(self.cfg.datasets[self.name], 'columns', None)
		if columns is None:
			df = read_csv(raw_path, dtype=str)
		else:
			df = read_csv(raw_path, usecols=columns, dtype=str)
			# Sanity Check
			for col in columns:
				if col not in df.columns:
					raise KeyError(f"Missing required column '{col}' in {raw_path}. Found: {list(df.columns)}")

		# Clean dataframe [Subsets, Dtypes, Drop Nans, Sort]
		# Should be done in the child class
		df = self._clean_raw_csv(df=df)
		return df

	def _clean_raw_csv(self, df: DataFrame):
		raise NotImplementedError('Should be implemented by child dataset')

	def _resolve_paths(self):
		# Base Dataset Dir
		raw_path = Path(join(self.cfg.runner.dirs.raw_datasets, f'{self.name}.csv'))

		# Refactored Paths
		ref_dir = Path(self.cfg.runner.dirs.refactored_datasets)
		ref_dir.mkdir(parents=True, exist_ok=True)
		# Paths
		ref_meta = Path(join(ref_dir, f'{self.name}.csv'))
		ref_npz = Path(join(ref_dir, f'{self.name}.npz'))
		# Raw Standardizer
		ref_std = Path(join(ref_dir, f'{self.name}.std.npz'))
		
		# Synthetic Paths
		syn_dir = Path(self.cfg.runner.dirs.synth_datasets)
		syn_dir.mkdir(parents=True, exist_ok=True)
		# Suffix for Synthetic Dataset [Number of samples]
		n_syn = self.cfg.runner.dataset.synth_samples
		syn_suffix = f'-N{int(n_syn)}' if (n_syn is not None) else ''
		# Paths
		syn_meta = Path(join(syn_dir, f'{self.name}{syn_suffix}.csv'))
		syn_npz = Path(join(syn_dir, f'{self.name}{syn_suffix}.npz'))
		# Synthetic Standardizer
		derived_std = Path(join(syn_dir, f'{self.name}.derived.std.npz'))

		return {'raw': raw_path,
				'ref_meta': ref_meta, 'ref_npz': ref_npz,
				'syn_meta': syn_meta, 'syn_npz': syn_npz,
				'ref_std': ref_std, 'derived_std': derived_std}

	def balance_classes(self):
		rng = random.default_rng(self.cfg.runner.seed)

		idx0 = flatnonzero(self.y == 0)
		idx1 = flatnonzero(self.y == 1)
		idx2 = flatnonzero(self.y == 2)

		n0, n1, n2 = len(idx0), len(idx1), len(idx2)
		if min(n0, n1, n2) == 0:
			raise RuntimeError(f"At least one class is empty: keep={n0} left={n1} right={n2}")

		target_n = min(n0, n1, n2)  # exact 1/3 balance
		self.log.info(f'Number of samples per class = {target_n} from {n0, n1, n2}')
		s0 = rng.choice(idx0, size=target_n, replace=False)
		s1 = rng.choice(idx1, size=target_n, replace=False)
		s2 = rng.choice(idx2, size=target_n, replace=False)

		sel = concatenate([s0, s1, s2])
		rng.shuffle(sel)

		self.X = self.X[sel]
		self.y = self.y[sel]
		self.meta = self.meta.iloc[sel].reset_index(drop=True)

		counts = bincount(self.y.astype(int64), minlength=3)
		self.log.info(
			f"[{self.name}] Balanced to min class: target_n={target_n} "
			f"original_counts={[n0, n1, n2]} new_counts={counts.tolist()}"
		)
		return self.X, self.y, self.meta

	def save(self):
		path_prefix = 'ref' if self.raw else 'syn'
		if self.meta is None or self.X is None or self.y is None:
			raise RuntimeError(f'Unable to save {self.name} dataset because X or y or meta are None')
		self.meta.to_csv(self.paths[f'{path_prefix}_meta'], index=False)
		savez_compressed(self.paths[f'{path_prefix}_npz'], X=self.X, y=self.y)

	def analyze(self):
		pass

	def __len__(self):
		return self.X.shape[0]