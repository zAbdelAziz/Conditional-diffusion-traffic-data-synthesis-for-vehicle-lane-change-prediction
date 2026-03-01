from os import cpu_count, replace
from os.path import join
from pathlib import Path

from numpy import random, arange, unique, int64

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch import device, cuda, Generator, save, load
from torch.utils.data import DataLoader, Subset

import wandb

from datasets.base import BaseDataset
from models.base import BaseModel

from utils.logger import Logger
from utils.config import Config
from utils.dataset_split import split_by_group, split_random


class BaseTrainer:
	def __init__(self, name: str, train_dataset: BaseDataset, model: BaseModel, test_dataset: BaseDataset = None):
		self.name = name

		self.cfg = Config()

		self.log = Logger(name=self.cfg.runner.name)

		self.device = device('cuda' if cuda.is_available() else 'cpu')

		self.main_dataset = train_dataset
		self.test_dataset = test_dataset

		# Analyze Dataset
		if self.cfg.runner.dataset.analyze:
			self.main_dataset.analyze()
			if self.test_dataset is not None:
				self.test_dataset.analyze()

		self.train_dataset, self.valid_dataset = None, None
		self.train_loader, self.valid_loader, self.test_loader = None, None, None

		# Split Dataset
		self._split_train_dataset()
		
		# Data Loaders
		self._build_loaders()
		
		# Model
		self.model = model.to(self.device)
		
		# Loss Function
		self.loss_fn = self._build_loss_fn().to(self.device)
		
		# Optimizer
		self.optimizer = self._build_optimizer()
		
		# LR Scheduler
		self.scheduler = self._build_scheduler()
		
		# Wandb
		self._init_wandb()
		
		# Best Model
		self._init_checkpointing()

	def _split_train_dataset(self):
		# Default Splitting Mode
		mode = self.cfg.trainers[self.name].split.mode
		p_train, p_valid, p_test = self.cfg.trainers[self.name].split.train, self.cfg.trainers[self.name].split.valid, self.cfg.trainers[self.name].split.test

		# Whether to ignore splitting test set
		ignore_test = True if self.test_dataset is not None else False

		# normalize proportions
		s = p_train + p_valid + p_test
		if abs(s - 1.0) > 1e-6:
			raise ValueError(f'Split ratios must sum to 1.0, got {s} (train={p_train}, valid={p_valid}, test={p_test})')

		rng = random.default_rng(self.cfg.runner.seed)
		idx = arange(len(self.main_dataset), dtype=int64)

		# Split by Group
		if mode == 'group':
			# Get the Unique Vehicle ids from meta
			vids = self.main_dataset.meta['Vehicle_ID'].to_numpy(int64)
			uniq = unique(vids)
			# Shuffle them
			rng.shuffle(uniq)
			
			# Split By Group [Vehicle]
			train_idx, valid_idx, test_idx, idx = split_by_group(p_train=p_train, p_valid=p_valid, p_test=p_test,
																 uniq=uniq, idx=idx, vids=vids, ignore_test=ignore_test)
		# Split Randomly
		elif mode == 'random':
			train_idx, valid_idx, test_idx = split_random(p_train=p_train, p_valid=p_valid, p_test=p_test,
														  n=len(self.main_dataset), rng=rng, ignore_test=ignore_test)
		else:
			raise KeyError(f'Split mode {mode} in {self.name} is not supported only group or random')

		# Create Subsets
		self.train_dataset = Subset(self.main_dataset, train_idx.tolist())
		self.valid_dataset = Subset(self.main_dataset, valid_idx.tolist())
		if self.test_dataset is None:
			self.test_dataset = Subset(self.main_dataset, test_idx.tolist())

		self.log.info(
			f'Split mode={mode}: train={len(self.train_dataset)} valid={len(self.valid_dataset)} test={len(self.test_dataset) if self.test_dataset is not None else 0}')

	def _build_loaders(self):
		g = Generator()
		# Data Loader Configuration
		loader_cfg = {'batch_size': self.cfg.trainers[self.name].batch_size, 'generator': g, 'pin_memory': True,
					  'num_workers': min(8, cpu_count() or 1), 'persistent_workers': True, 'prefetch_factor': 4}
		# Loaders
		self.train_loader = DataLoader(self.train_dataset, shuffle=True, **loader_cfg)
		self.valid_loader = DataLoader(self.valid_dataset, shuffle=False, **loader_cfg)
		self.test_loader = DataLoader(self.test_dataset, shuffle=False, **loader_cfg)

	def _build_loss_fn(self):
		loss_name = self.cfg.trainers[self.name].loss.clsName
		try:
			loss_fn = getattr(nn, loss_name)
		except AttributeError:
			raise ValueError(f"Unknown loss function: {loss_name} for {self.name} Trainer")
		return loss_fn()

	def _build_optimizer(self):
		optim_name = self.cfg.trainers[self.name].optimizer.clsName
		optim_params = self.cfg.trainers[self.name].optimizer.params
		try:
			optimizer_fn = getattr(optim, optim_name)
		except AttributeError:
			raise ValueError(f"Unknown optimizer function: {optim_name} for {self.name} Trainer")
		return optimizer_fn(self.model.parameters(), **optim_params)

	def _build_scheduler(self):
		# OPTIONAL scheduler config
		tcfg = self.cfg.trainers[self.name]
		scfg = getattr(tcfg, "scheduler", None)
		if scfg is None:
			self.log.info("No scheduler configured.")
			return None

		# allow disabling via enabled flag
		if hasattr(scfg, "enabled") and not bool(scfg.enabled):
			self.log.info("Scheduler disabled via config.")
			return None

		sched_name = getattr(scfg, "clsName", None)
		if sched_name is None:
			self.log.info("Scheduler config present but no clsName; skipping.")
			return None

		params = dict(getattr(scfg, "params", {}) or {})

		# Resolve common cosine defaults
		if sched_name == "CosineAnnealingLR":
			# If user didn't set T_max, default to epochs when stepping per-epoch
			step_per = str(getattr(scfg, "step_per", "epoch")).lower()
			if "T_max" not in params:
				if step_per == "epoch":
					params["T_max"] = int(self.cfg.trainers[self.name].epochs)
				else:
					# per-step cosine: set later once loaders exist
					# Here we set a placeholder; DownstreamTrainer can overwrite.
					params["T_max"] = int(self.cfg.trainers[self.name].epochs)

		try:
			sched_cls = getattr(lr_scheduler, sched_name)
		except AttributeError:
			raise ValueError(f"Unknown scheduler: {sched_name}. Must exist in torch.optim.lr_scheduler")

		sch = sched_cls(self.optimizer, **params)
		self.log.info(f"Built scheduler: {sched_name} params={params}")
		return sch

	def start(self):
		raise NotImplementedError

	def _init_wandb(self):
		self.wandb_run = None
		wb_cfg = getattr(self.cfg, "wandb", None)
		if wb_cfg is None or not bool(getattr(wb_cfg, "enabled", False)):
			return

		d = "-d@" if self.name == 'downstream' else '-d'
		s = "-s@" if self.name == 'diffSynth' else '-s'
		run_name = f"{self.cfg.runner.name}{d if self.cfg.runner.train.downstream else ""}{s if self.cfg.runner.train.synth else ""}"
		self.wandb_run = wandb.init(
			project=getattr(wb_cfg, "project", None),
			entity=getattr(wb_cfg, "entity", None),
			name=run_name,
			tags=list(getattr(wb_cfg, "tags", [])) if getattr(wb_cfg, "tags", None) else None,
			notes=getattr(wb_cfg, "notes", ""),
			config=self.cfg.as_dict(),
			reinit=True
		)

		watch_cfg = getattr(wb_cfg, "watch", None)
		if watch_cfg is not None and bool(getattr(watch_cfg, "enabled", False)):
			wandb.watch(
				self.model,
				log=getattr(watch_cfg, "log", "gradients"),
				log_freq=int(getattr(watch_cfg, "log_freq", 100)),
			)

		# Epoch-only x-axis
		wandb.define_metric("epoch")
		wandb.define_metric("train/*", step_metric="epoch")
		wandb.define_metric("valid/*", step_metric="epoch")
		wandb.define_metric("test/*", step_metric="epoch")

		self.log.info(f"W&B initialized: {self.wandb_run.name}")

	def _wandb_log(self, data: dict, epoch: int):
		if self.wandb_run is None:
			return
		# enforce epoch-only
		data = dict(data)
		data["epoch"] = int(epoch)
		wandb.log(data, step=int(epoch))

	def _init_checkpointing(self):
		self.best_metric = None
		self.best_epoch = -1

		# Run name used for folder/file naming (wandb-like)
		self.run_name = None
		if self.wandb_run is not None:
			self.run_name = self.wandb_run.name
		else:
			# fallback to run_name logic
			self.run_name = f"{self.cfg.runner.name}{'-d' if self.cfg.runner.train.downstream else ''}{'-s' if self.cfg.runner.train.synth else ''}"

		# choose root dir based on trainer type (downstream vs synth)
		root = self.cfg.runner.dirs.root_models
		if self.name == "downstream":
			base_dir = self.cfg.runner.dirs.downstream_model
		elif self.name == "diffSynth":
			base_dir = self.cfg.runner.dirs.synth_model
		else:
			base_dir = join(Path(root), self.run_name)

		self.ckpt_dir = Path(join(Path(base_dir), self.run_name))
		self.ckpt_dir.mkdir(parents=True, exist_ok=True)

		self.log.info(f"Checkpoint dir: {self.ckpt_dir}")

	def save_checkpoint(self, epoch: int, metric_name: str, metric_value: float, is_best: bool, tag: str = None):
		"""
		Save a checkpoint payload
			'tag' can be "best" or "last" etc.
		Trainer decides when to call with is_best=True.
		"""
		payload = {
			"epoch": int(epoch),
			"metric_name": str(metric_name),
			"metric_value": float(metric_value),
			"model_state": self.model.state_dict(),
			"optimizer_state": self.optimizer.state_dict(),
			"scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
			"cfg": self.cfg.as_dict(),
			"run_name": self.run_name,
		}

		# filename
		tag = tag or ("best" if is_best else "ckpt")
		fname = f"{self.run_name}-{tag}.pt"
		path = self.ckpt_dir / fname

		# atomic-ish write: write tmp then replace
		tmp_path = self.ckpt_dir / (fname + ".tmp")
		save(payload, tmp_path)
		replace(tmp_path, path)

		# update best bookkeeping if needed
		if is_best:
			self.best_metric = float(metric_value)
			self.best_epoch = int(epoch)

			# convenient stable pointer file
			best_link = self.ckpt_dir / f"{self.run_name}-best.pt"
			if best_link != path:
				# copy/replace pointer
				save(payload, tmp_path)
				replace(tmp_path, best_link)

		self.log.info(
			f"Saved checkpoint: {path.name} "
			f"(epoch={epoch}, {metric_name}={metric_value:.6f}, best={is_best})"
		)

		return str(path)

	def load_checkpoint(self, path: str, load_optimizer: bool = False, load_scheduler: bool = False):
		"""
		Loads a checkpoint into the current trainer's model (and optionally optimizer/scheduler)
		Returns the checkpoint dict metadata (epoch, metric, etc..)
		"""
		path = str(path)
		ckpt = load(path, map_location=self.device)

		self.model.load_state_dict(ckpt["model_state"], strict=True)

		if load_optimizer and ckpt.get("optimizer_state") is not None:
			self.optimizer.load_state_dict(ckpt["optimizer_state"])

		if load_scheduler and self.scheduler is not None and ckpt.get("scheduler_state") is not None:
			self.scheduler.load_state_dict(ckpt["scheduler_state"])

		self.log.info(
			f"Loaded checkpoint: {Path(path).name} "
			f"(epoch={ckpt.get('epoch')}, {ckpt.get('metric_name')}={ckpt.get('metric_value')})"
		)
		return ckpt

	@property
	def best_checkpoint_path(self) -> str:
		"""
		Returns the stable 'best' pointer path if it exists, else None
		"""
		p = Path(join(self.ckpt_dir, f"{self.run_name}-best.pt"))
		return str(p) if p.exists() else None

	@property
	def _current_lr(self):
		# helper for logging
		return float(self.optimizer.param_groups[0]["lr"])