from pathlib import Path
from typing import Union
from dataclasses import dataclass

from omegaconf import OmegaConf, DictConfig


@dataclass
class ConfigStruct:
	runner: DictConfig
	datasets: DictConfig
	models: DictConfig
	trainers: DictConfig
	logger: DictConfig
	wandb: DictConfig


class Config(ConfigStruct):
	def __init__(self, path: Union[str, Path, None] = None, *, resolve: bool = True):
		# Default Path
		if path is None:
			path = "config.yaml"

		path = Path(path)
		if not path.exists():
			raise FileNotFoundError(f"Config file not found: {path}")

		# Load into DictConfig
		self.cfg: DictConfig = OmegaConf.load(path)

		# Resolve interpolations immediately
			# if False: interpolations stay lazy until accessed/converted
		if resolve:
			OmegaConf.resolve(self.cfg)

		# Expose top-level keys directly on the object
		for k in self.cfg.keys():
			setattr(self, k, self.cfg[k])

	def as_dict(self):
		# Return a plain dict with interpolations resolved
		return OmegaConf.to_container(self.cfg, resolve=True)

	def as_yaml(self):
		# Return YAML string
		return OmegaConf.to_yaml(self.cfg, resolve=True)

	def __getitem__(self, item):
		return getattr(self, item)
