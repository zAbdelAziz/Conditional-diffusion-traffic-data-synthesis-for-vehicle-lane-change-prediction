from numpy import load, array, float32, int32, savez_compressed

from numpy import concatenate
from torch import no_grad
from torch.utils.data import Subset
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm

import wandb

from trainers.base import BaseTrainer
from datasets.ngsim.downstream_feature_builder import DownstreamFeatureBuilder
from utils.standardizer import DownstreamStandardizer
from utils.metrics import *


class DownstreamTrainer(BaseTrainer):
	def __init__(self, train_dataset, model, test_dataset=None):
		super().__init__(name="downstream", train_dataset=train_dataset, test_dataset=test_dataset, model=model)

		# Build Downstream Features
		self._build_features()

		# Rebuild loaders since BaseTrainer built them before caching
		self._build_loaders()

		# After X becomes (N,T,61)
		if self.cfg.trainers.downstream.standardize:
			self._build_standardizer()

		self.global_step = 0

	def start(self):
		for epoch in range(self.cfg.trainers.downstream.epochs):
			train_loss, train_metrics = self._train_epoch()
			val_loss, val_metrics = self._eval_epoch(self.valid_loader)

			self._log_epoch("train", epoch, train_loss, train_metrics)
			self._log_epoch("valid", epoch, val_loss, val_metrics)

			self._save_best(epoch, val_loss, val_metrics)

		# Load Best model
		self.load_checkpoint(self.best_checkpoint_path, load_optimizer=False, load_scheduler=False)

		test_loss, test_metrics = self._eval_epoch(self.test_loader)
		self._log_epoch("test", self.cfg.trainers.downstream.epochs, test_loss, test_metrics)
	
	def _train_epoch(self):
		self.model.train()
		total_loss = 0.0
		total = 0

		y_true_all = []
		y_pred_all = []

		for step, (X, y) in enumerate(tqdm(self.train_loader, desc="train")):
			loss, logits, total, total_loss = self.predict(X, y, total, total_loss)

			loss.backward()

			if self.cfg.trainers.downstream.clip_grad:
				clip_grad_norm_(self.model.parameters(), max_norm=1.0)

			self.optimizer.step()

			pred = logits.argmax(dim=1)
			y_true_all.append(y.detach().cpu().numpy())
			y_pred_all.append(pred.detach().cpu().numpy())

		# Step Scheduler
		if self.scheduler is not None:
			self.scheduler.step()

		avg_loss = total_loss / max(total, 1)
		y_true = concatenate(y_true_all)
		y_pred = concatenate(y_pred_all)
		metrics = self.compute_classification_metrics(y_true, y_pred, num_classes=3)

		return avg_loss, metrics

	def _eval_epoch(self, loader):
		self.model.eval()
		total_loss = 0.0
		total = 0

		y_true_all = []
		y_pred_all = []

		with no_grad():
			for X, y in tqdm(loader, desc="eval"):
				loss, logits, total, total_loss = self.predict(X, y, total, total_loss)

				pred = logits.argmax(dim=1)
				y_true_all.append(y.detach().cpu().numpy())
				y_pred_all.append(pred.detach().cpu().numpy())

		avg_loss = total_loss / max(total, 1)
		y_true = concatenate(y_true_all)
		y_pred = concatenate(y_pred_all)
		metrics = self.compute_classification_metrics(y_true, y_pred, num_classes=3)

		# Log Confusion if Test set
		if loader == self.test_loader and self.wandb_run is not None:
			cm_plot = wandb.plot.confusion_matrix(
				y_true=y_true.tolist(),
				preds=y_pred.tolist(),
				class_names=["0", "1", "2"],
			)
			self._wandb_log({"test/confusion_matrix": cm_plot}, epoch=self.cfg.trainers.downstream.epochs)

		return avg_loss, metrics

	@staticmethod
	def compute_classification_metrics(y_true: ndarray, y_pred: ndarray, num_classes: int = 3):
		cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)

		out = {
			"cm": cm,
			"acc": accuracy_from_cm(cm),
			"balanced_acc": balanced_accuracy_from_cm(cm),
			"macro_f1": macro_f1_from_cm(cm),
			"weighted_f1": weighted_f1_from_cm(cm),
			"mcc": mcc_from_cm(cm),
		}

		return out

	def _load_best(self):
		# best_path = ()
		if self.best_checkpoint_path is None:
			self.log.warning("No best checkpoint found; evaluating test with current (last) model.")
		else:
			# only need model weights for evaluation
			self.load_checkpoint(self.best_checkpoint_path, load_optimizer=False, load_scheduler=False)

		# log best metadata to wandb
		if self.wandb_run is not None and self.best_checkpoint_path is not None:
			self._wandb_log(
				{
					"best/epoch": int(self.best_epoch),
					"best/metric": float(self.best_metric),
					"best/path": str(self.best_checkpoint_path),
				},
				epoch=self.cfg.trainers.downstream.epochs,
			)

	def _save_best(self, epoch, val_loss, val_metrics):
		cfg = self.cfg.trainers.downstream.checkpoint
		metric_key = cfg.metric

		if cfg.enabled:
			# map metric_key -> value
			metrics_map = {
				"valid/loss": float(val_loss),
				"valid/acc": float(val_metrics["acc"]),
				"valid/macro_f1": float(val_metrics["macro_f1"]),
				"valid/balanced_acc": float(val_metrics["balanced_acc"]),
				"valid/weighted_f1": float(val_metrics["weighted_f1"]),
				"valid/mcc": float(val_metrics["mcc"]),
			}
			if metric_key not in metrics_map:
				raise ValueError(f"Unknown checkpoint metric '{metric_key}'. Supported: {list(metrics_map.keys())}")

			score = metrics_map[metric_key]

			is_best = False
			if self.best_metric is None:
				is_best = True
			else:
				if cfg.mode == "min":
					is_best = score < self.best_metric
				elif cfg.mode == "max":
					is_best = score > self.best_metric
				else:
					raise ValueError("checkpoint.mode must be 'min' or 'max'")

			if is_best:
				self.save_checkpoint(epoch=epoch, metric_name=metric_key, metric_value=score, is_best=True, tag="best")

	def _log_epoch(self, split: str, epoch: int, loss: float, metrics: dict):
		self.log.info(
			f"{split.upper()} epoch={epoch:03d} "
			f"loss={loss:.4f} acc={metrics['acc']:.3f} "
			f"macro_f1={metrics['macro_f1']:.3f} balanced_acc={metrics['balanced_acc']:.3f} "
			f"mcc={metrics['mcc']:.3f}"
		)

		wb = {
			f"{split}/loss": float(loss),
			f"{split}/acc": float(metrics["acc"]),
			f"{split}/balanced_acc": float(metrics["balanced_acc"]),
			f"{split}/macro_f1": float(metrics["macro_f1"]),
			f"{split}/weighted_f1": float(metrics["weighted_f1"]),
			f"{split}/mcc": float(metrics["mcc"]),
			f"{split}/lr": float(self._current_lr),
		}

		self._wandb_log(wb, epoch=epoch)

	def predict(self, X, y, total, total_loss):
		X, y = X.to(self.device), y.to(self.device)

		logits = self.model(X)
		loss = self.loss_fn(logits, y)

		bs = X.size(0)
		total_loss += loss.item() * bs
		total += bs
		return loss, logits, total, total_loss

	def _build_features(self):
		"""One-shot derivation cache for downstream"""
		# Get Vision R
		vision_R = getattr(self.main_dataset, "vision_R", None)
		if vision_R is None:
			try:
				std = load(self.main_dataset.paths["ref_std"])
				vision_R = float(std["vision_R"])
				self.main_dataset.vision_R = vision_R
			except Exception as e:
				raise ValueError("vision_R not available cant derive downstream features") from e
		# Get dt
		dt = self.cfg.datasets[self.main_dataset.name].preprocessing.dt
		builder = DownstreamFeatureBuilder(dt=dt, vision_R=self.main_dataset.vision_R)

		# Cache derived features for the MAIN dataset
		Xd_main = builder.build(self.main_dataset.X, batch_size=self.cfg.trainers.downstream.batch_size)
		self.main_dataset.X = Xd_main
		self.main_dataset.D = int(Xd_main.shape[-1])  # 38

		# If test_dataset is external (not a Subset of main) cache it too
		# external test_dataset remains as provided (BaseDataset), not Subset(main)
		# internal test is Subset(main)
		# STUPID but OK
		if self.test_dataset is not None:
			# Determine whether test is Subset(main_dataset) or external
			is_subset = isinstance(self.test_dataset, Subset)
			if is_subset:
				# its already backed by main_dataset, so done
				pass
			else:
				# external: must derive once as well
				if not hasattr(self.test_dataset, "X") or self.test_dataset.X is None:
					raise RuntimeError("External test_dataset has no .X to derive from.")
				if int(getattr(self.test_dataset, "D", self.test_dataset.X.shape[-1])) != 14:
					raise RuntimeError(
						f"Expected external test_dataset D=20, got D={self.test_dataset.X.shape[-1]}")

				Xd_test = builder.build(self.test_dataset.X, batch_size=self.cfg.trainers.downstream.batch_size)
				self.test_dataset.X = Xd_test
				self.test_dataset.D = int(Xd_test.shape[-1])

		self.log.info(f'[downstream] Cached derived features: main X={self.main_dataset.X.shape} '
					  f'{"| external test cached" if (self.test_dataset is not None and not isinstance(self.test_dataset, Subset)) else ""}')

	def _build_standardizer(self):
		self.standardizer = DownstreamStandardizer()
		train_idx = array(self.train_dataset.indices, dtype=int64)
		self.std_mu, self.std_sigma = self.standardizer.fit(self.main_dataset.X, train_idx)

		self.main_dataset.X = self.standardizer.transform(self.main_dataset.X, self.std_mu, self.std_sigma)

		# external test dataset
		if self.test_dataset is not None and not isinstance(self.test_dataset, Subset):
			self.test_dataset.X = self.standardizer.transform(self.test_dataset.X, self.std_mu, self.std_sigma)

		savez_compressed(self.main_dataset.paths['derived_std'],
						 mu=self.std_mu.astype(float32), sigma=self.std_sigma.astype(float32),
						 vision_R=self.main_dataset.vision_R, D=int32(self.std_mu.shape[0]))