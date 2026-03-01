from pandas import DataFrame
from numpy import array, random, full, arange, concatenate, empty, zeros, load, savez_compressed, int32, int64, float32

from torch.utils.data import Subset
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch import compile, randint, randn_like, long, no_grad, inference_mode, from_numpy, cuda

from tqdm import tqdm

from trainers.base import BaseTrainer
from utils.standardizer.diffusion_standardizer import DiffusionStandardizer


class SynthTrainer(BaseTrainer):
	def __init__(self, train_dataset, model, diffusion, test_dataset=None, y_conditional: bool = True):
		super().__init__(name="diffSynth", train_dataset=train_dataset, test_dataset=test_dataset, model=model)

		# Standardize Dataset
		self._build_standardizer()

		self.epochs = self.cfg.trainers.diffSynth.epochs
		
		self.diffusion = diffusion.to(self.device)
		self.global_step = 0

		self.clip_grad = bool(getattr(self.cfg.trainers.diffSynth, "clip_grad", False))
		# conditional diffusion
		self.conditional = y_conditional
		
		self.loss_fn = mse_loss

	def start(self):
		for epoch in range(self.epochs):
			train_loss, train_metrics = self._train_epoch()
			val_loss, val_metrics = self._eval_epoch(self.valid_loader)

			self._log_epoch("train", epoch, train_loss, train_metrics)
			self._log_epoch("valid", epoch, val_loss, val_metrics)

			self._save_best(epoch, val_loss)

		# Load best model for test
		if self.best_checkpoint_path is not None:
			self.load_checkpoint(self.best_checkpoint_path, load_optimizer=False, load_scheduler=False)

		test_loss, test_metrics = self._eval_epoch(self.test_loader)
		self._log_epoch("test", self.epochs, test_loss, test_metrics)

		try:
			if self.cfg.trainers.diffSynth.compile:
				self.model = compile(self.model, mode="reduce-overhead")
		except:
			pass

	def _train_epoch(self):
		self.model.train()
		total_loss = 0.0
		total = 0

		for X, y in tqdm(self.train_loader, desc="Train Synth"):
			X, y = X.to(self.device), y.to(self.device)
			
			# Sample x_t, eps, t
			x_t, t, eps = self._sample(X)
			
			# Predict eps_hat
			eps_hat = self._predict(x_t, t, y)
			
			# Zero Grad
			self.optimizer.zero_grad()
			
			# Loss
			loss, total, total_loss = self._calc_loss(eps_hat=eps_hat, eps=eps, total=total, total_loss=total_loss, batch_size=X.size(0))
			loss.backward()
			
			# Clip Gradient
			if self.clip_grad:
				clip_grad_norm_(self.model.parameters(), max_norm=1.0)
			
			# Optimizer Step
			self.optimizer.step()

			self.global_step += 1
		# Scheduler Step per-epoch
		if self.scheduler is not None:
			step_per = str(getattr(self.cfg.trainers.diffSynth.scheduler, "step_per", "epoch")).lower()
			if step_per == "epoch":
				self.scheduler.step()
		avg_loss = total_loss / max(total, 1)
		return avg_loss, {"loss": avg_loss}

	@no_grad()
	def _eval_epoch(self, loader):
		self.model.eval()
		total_loss = 0.0
		total = 0

		for X, y in tqdm(loader, desc="eval-synth"):
			X, y = X.to(self.device), y.to(self.device)

			# Sample x_t, eps, t
			x_t, t, eps = self._sample(X)

			# Predict eps_hat
			eps_hat = self._predict(x_t, t, y)
			
			# Loss
			loss, total, total_loss = self._calc_loss(eps_hat=eps_hat, eps=eps, total=total, total_loss=total_loss, batch_size=X.size(0))

		avg_loss = total_loss / max(total, 1)
		return avg_loss, {"loss": avg_loss}
		
	def _sample(self, X):
		# [B,T,D]
		t = randint(0, self.diffusion.T, (X.size(0),), device=self.device, dtype=long)
		
		eps = randn_like(X)
		x_t = self.diffusion.q_sample(X, t, eps)
		return x_t, t, eps
	
	def _predict(self, x_t, t, y):
		if self.conditional:
			return self.model(x_t, t, y)
		else:
			return self.model(x_t, t)
	
	def _calc_loss(self, eps_hat, eps, total_loss: float, total: int, batch_size):
		loss = self.loss_fn(eps_hat, eps)
		total_loss += float(loss.item()) * batch_size
		total += batch_size
		return loss, total, total_loss

	def _save_best(self, epoch, val_loss):
		if not self.cfg.trainers.diffSynth.checkpoint.enabled:
			return
		
		metric_key = str(self.cfg.trainers.diffSynth.checkpoint.metric)
		metrics_map = {"valid/loss": float(val_loss),}
		
		if metric_key not in metrics_map:
			raise ValueError(f"Unknown checkpoint metric '{metric_key}'. Supported: {list(metrics_map.keys())}")
		
		score = metrics_map[metric_key]

		is_best = False
		if self.best_metric is None:
			is_best = True
		else:
			if self.cfg.trainers.diffSynth.checkpoint.mode == "min":
				is_best = score < self.best_metric
			elif self.cfg.trainers.diffSynth.checkpoint.mode == "max":
				is_best = score > self.best_metric
			else:
				raise ValueError("checkpoint.mode must be 'min' or 'max'")
		if is_best:
			self.save_checkpoint(epoch=epoch, metric_name=metric_key, metric_value=score, is_best=True, tag="best")

	def _log_epoch(self, split: str, epoch: int, loss: float, metrics: dict):
		self.log.info(f"{split.upper()} epoch={epoch:03d} mse={loss:.6f}")
		wb = {
			f"{split}/loss": float(loss),
			f"{split}/lr": float(self._current_lr),
		}
		self._wandb_log(wb, epoch=epoch)

	@no_grad()
	def generate_synthetic(self, num_samples: int):
		self.model.eval()
		self.diffusion.eval()

		T = int(self.main_dataset.T)
		D = int(self.main_dataset.D)

		# Labels
		rng = random.default_rng(self.cfg.runner.seed)
		if self.conditional:
			if self.cfg.trainers.diffSynth.balance_labels:
				# Number of samples per class
				per = num_samples // self.model.num_classes
				# Full Matrix
				ys = [full((per,), c, dtype=int64) for c in range(self.model.num_classes)]
				# Remainder
				rem = num_samples - per * self.model.num_classes
				if rem > 0:
					# Append Remainder to first class
					ys.append(rng.integers(0, self.model.num_classes, size=(rem,), dtype=int64))
				y = concatenate(ys, axis=0)
				# Shuffle everything
				rng.shuffle(y)
			else:
				y = rng.integers(0, self.model.num_classes, size=(num_samples,), dtype=int64)
		else:
			y = rng.integers(0, self.model.num_classes, size=(num_samples,), dtype=int64)
		y_out = y.astype(int64, copy=False)

		# Meta
		meta = DataFrame({"Vehicle_ID": arange(num_samples, dtype=int64), "End_Frame_ID": zeros((num_samples,), dtype=int64), "y": y_out})

		# Inputs
		# tqdm over total samples
		pbar = tqdm(total=num_samples, desc="Generate Synthetic Data", unit="sample")
		# Empty X
		X_out = empty((num_samples, T, D), dtype=float32)

		start = 0
		try:
			with inference_mode():
				while start < num_samples:
					end = min(start + self.cfg.trainers.diffSynth.batch_size, num_samples)
					bs = end - start

					# Conditional
					y_bs = from_numpy(y_out[start:end]).to(self.device) if self.conditional else None
					shape = (bs, T, D)

					# Sample X_t from diffusion model
					X_t = self.diffusion.p_sample_loop(self.model, shape=shape, y=y_bs,
													   steps=self.cfg.trainers.diffSynth.sampling.steps,
													   method=self.cfg.trainers.diffSynth.sampling.method,
													   eta=self.cfg.trainers.diffSynth.sampling.eta)
					# Move results from GPU to CPU
					X_out[start:end] = X_t.detach().float().cpu().numpy()

					# free GPU memory between chunks
					# The memory overflows if generating a full dataset on the GPU at once
					del X_t, y_bs
					if self.device.type == "cuda":
						cuda.empty_cache()
					start = end
					pbar.update(bs)
		finally:
			pbar.close()

		return X_out, y_out, meta

	def _build_standardizer(self):
		try:
			# Try loading reference STD
			std = load(self.main_dataset.paths['ref_std'])
			# Vision R
			self.main_dataset.vision_R = std['vision_R'].astype(float32)
			self.std_mu = std['mu'].astype(float32)
			self.std_sigma = std['sigma'].astype(float32)
			self.standardizer = DiffusionStandardizer(vision_R=self.main_dataset.vision_R)
		except Exception:
			# Otherwise fit from TRAIN indices only
			self.log.error('Standardizer not found, generating values')
			train_idx = array(self.train_dataset.indices, dtype=int64)
			self.standardizer = DiffusionStandardizer(vision_R=self.main_dataset.vision_R)
			self.std_mu, self.std_sigma = self.standardizer.fit(self.main_dataset.X, train_idx)

			savez_compressed(
				self.main_dataset.paths['ref_std'],
				mu=self.std_mu.astype(float32),
				sigma=self.std_sigma.astype(float32),
				vision_R=self.main_dataset.vision_R,
				D=int32(self.std_mu.shape[0]),
			)

		# Standardize MAIN dataset in-place (Subsets will reflect this)
		self.main_dataset.X = self.standardizer.transform(self.main_dataset.X, self.std_mu, self.std_sigma)

		# Standardize external test dataset ONLY if it's a real dataset with X (not a Subset of main)
		if self.test_dataset is not None:
			if isinstance(self.test_dataset, Subset):
				# If subset of main_dataset, nothing to do (already standardized above)
				if self.test_dataset.dataset is self.main_dataset:
					return
				# If subset of some other BaseDataset, transform its underlying dataset once
				base = self.test_dataset.dataset
				if hasattr(base, "X"):
					base.X = self.standardizer.transform(base.X, self.std_mu, self.std_sigma)
				return

			# Non-subset dataset: standardize normally
			if hasattr(self.test_dataset, "X"):
				self.test_dataset.X = self.standardizer.transform(self.test_dataset.X, self.std_mu, self.std_sigma)

