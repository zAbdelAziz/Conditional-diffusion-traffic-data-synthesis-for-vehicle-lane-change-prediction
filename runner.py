import inspect
import numpy as np

from omegaconf import DictConfig

import datasets
import models
from trainers.synth import SynthTrainer
from trainers.downstream import DownstreamTrainer

from utils.config import Config
from utils.logger import Logger
from utils.seeder import set_global_seed


class Runner:
	def __init__(self):
		self.cfg = Config()

		self.log = Logger(self.cfg.runner.name)
		
		# Dataset/Model Names
		self.train_dataset_name = self.cfg.runner.dataset.train
		self.test_dataset_name = self.cfg.runner.dataset.test
		self.synth_model_name = self.cfg.runner.model.synth
		self.diffusion_model_name = self.cfg.runner.model.diffusion
		self.downstream_model_name = self.cfg.runner.model.downstream

		# Empty Datasets, Models, Trainers
		self.dataset, self.raw_dataset, self.decomposed_dataset = None, None, None
		self.synth_model, self.diff_model, self.synth_trainer = None, None, None
		self.downstream_model, self.downstream_trainer = None, None

		# Available [Dynamic] Classes
		self.available_dataset_classes = [name for name, _ in inspect.getmembers(datasets, inspect.isclass)]
		self.available_models = [name for name, _ in inspect.getmembers(models, inspect.isclass)]

	def run(self):
		# Set global seeds
		set_global_seed(int(self.cfg.runner.seed), deterministic=True)
		
		train_synth = self.cfg.runner.train.synth
		train_downstream = self.cfg.runner.train.downstream
		
		# Validate Dataset/Model Names/Classes exist
		# Just for Debugging
		self.validate_name(name=self.cfg.runner.dataset.train, obj=self.cfg.datasets, identifier='dataset', available_classes=self.available_dataset_classes)
		self.validate_name(name=self.cfg.runner.dataset.test, obj=self.cfg.datasets, identifier='dataset', available_classes=self.available_dataset_classes)
		self.validate_name(name=self.cfg.runner.model.synth, obj=self.cfg.models, identifier='model', available_classes=self.available_models)
		self.validate_name(name=self.cfg.runner.model.downstream, obj=self.cfg.models, identifier='model', available_classes=self.available_models)

		# TODO Online Synthesis [After thesis]
		if self.cfg.runner.dataset.online_synth:
			raise NotImplementedError('disable online synth [TODO after thesis]')

		self.log.info(f'Runner start\n\t'
					  f'Name: {self.cfg.runner.name}\n\tTrain Dataset: {self.cfg.runner.dataset.train}\n\tTest Dataset: {self.cfg.runner.dataset.test}\n\t===\n\t'
					  f'Train Synth: {train_synth}\n\tTrain Downstream: {train_downstream}\n\t===\n\t'
					  f'Synth Model: {self.cfg.runner.model.synth}\n\tDownstream Model: {self.cfg.runner.model.downstream}\n\t===\n\t'
					  f'Analyze Dataset: {self.cfg.runner.dataset.analyze}\n\tOnline Synth: {self.cfg.runner.dataset.online_synth}\n\t')
		
		# TODO Inference Style [After thesis]
		if not train_synth and not train_downstream:
			self.log.info('Nothing to run: train.synth=False and train.downstream=False')
			return
		
		if train_synth:
			self.train_synth()
		
		if train_downstream:
			self.train_downstream()
		
		# Shutdown logger
		self.log.info('Runner finished successfully')
		self.log.flush()

	def train_synth(self):
		self.log.info('SYNTH TRAINING')

		# Dataset
		self.log.info(f'Initializing RAW dataset: {self.train_dataset_name}')
		self.raw_dataset = getattr(datasets, self.cfg.datasets[self.train_dataset_name].clsName)(name=self.train_dataset_name, raw=True)

		# Noise Model
		self.log.info(f'Building Diffusion Noise model: {self.diffusion_model_name}')
		self.diff_model = getattr(models, self.cfg.models[self.diffusion_model_name].clsName)(**self.cfg.models[self.diffusion_model_name].hyperparams)
		self.log.info(self.diff_model)

		# Denoiser Model
		self.log.info(f'Building Diffusion Denoiser model: {self.diffusion_model_name}')
		self.synth_model = getattr(models, self.cfg.models[self.synth_model_name].clsName)(**self.cfg.models[self.synth_model_name].hyperparams)
		self.log.info(self.synth_model)

		# Synth Trainer
		self.log.info('Building Synth trainer')
		self.synth_trainer = SynthTrainer(train_dataset=self.raw_dataset, model=self.synth_model, diffusion=self.diff_model)

		self.log.info('Starting Synth trainer')
		self.synth_trainer.start()

		n_syn = self.cfg.runner.dataset.synth_samples
		if n_syn <= 0:
			raise ValueError('runner.dataset.synth_samples must be a positive integer to generate synthetic data')

		# Generate and Save Synthetic Dataset
		# self.log.info(f'Generating synthetic dataset: N={n_syn}')
		# X_syn, y_syn, meta_syn = self.synth_trainer.generate_synthetic(num_samples=n_syn)
		# self.save_synth_dataset(X=X_syn, y=y_syn, meta=meta_syn)

		# Use output as synthetic dataset [Technically just reload the class with raw=False]
		self.dataset = getattr(datasets, self.cfg.datasets[self.train_dataset_name].clsName)(name=self.train_dataset_name, raw=False)
		
		# Delete Raw Dataset [Memory issue]
		del self.raw_dataset
		self.raw_dataset = None
		
	def train_downstream(self):
		self.log.info('DOWNSTREAM TRAINING')
		if self.dataset is None:
			# Load [Train] Dataset if dataset is None
			self.log.info(f'Loading pre-generated dataset [{self.train_dataset_name}] from disk')
			self.dataset = getattr(datasets, self.cfg.datasets[self.train_dataset_name].clsName)(name=self.train_dataset_name, raw=False)

		# Initialize Test Dataset
		if self.train_dataset_name == self.test_dataset_name:
			self.log.info('Train/Test dataset names match -> training on SYNTH, testing on RAW')
			test_dataset = getattr(datasets, self.cfg.datasets[self.test_dataset_name].clsName)(name=self.test_dataset_name, raw=True)
		else:
			self.log.info('Train/Test dataset names differ -> training on SYNTH, testing on SYNTH')
			test_dataset = getattr(datasets, self.cfg.datasets[self.test_dataset_name].clsName)(name=self.test_dataset_name, raw=False)

		self.log.info('Building downstream Model')
		self.downstream_model = getattr(models, self.cfg.models[self.downstream_model_name].clsName)(**self.cfg.models[self.downstream_model_name].hyperparams)
	
		# TODO Pass Trainer Parameters
		self.log.info('Building downstream Trainer')
		self.downstream_trainer = DownstreamTrainer(train_dataset=self.dataset, test_dataset=test_dataset, model=self.downstream_model)
	
		self.log.info('Starting downstream training')
		self.downstream_trainer.start()


	def save_synth_dataset(self, X, y, meta):
		# De-Standardize X [Always save Raw-like Samples non-normalized and non-standardized]
		if self.cfg.trainers.diffSynth.standardize:
			std = self.synth_trainer.standardizer
			mu = self.synth_trainer.std_mu
			sigma = self.synth_trainer.std_sigma
			# Inverse Transform X
			if std is not None and mu is not None and sigma is not None:
				X = std.inverse_transform(X, mu, sigma)
			else:
				# Should Not Happen but just for debugging
				self.log.warning('Standardize enabled but no stored (std, mu, sigma) found so saving standardized synth as-is')

		# Write to the synthetic cache paths
		meta.to_csv(self.raw_dataset.paths['syn_meta'], index=False)
		np.savez_compressed(self.raw_dataset.paths['syn_npz'], X=X, y=y)
		self.log.info(f'Saved synthetic cache:\n\tmeta={self.raw_dataset.paths['syn_meta']}\n\tnpz={self.raw_dataset.paths['syn_npz']}')

	@staticmethod
	def validate_name(name: str, obj: DictConfig, identifier: str, available_classes: list):
		# Make sure object name exists with a valid class name
		if name not in obj.keys():
			raise KeyError(f'{identifier.upper()} {name} is not available in {identifier}s,\n\t'
						   f'in runner.{identifier}.name select one of {obj.keys()}')
		if 'clsName' not in obj[name].keys():
			raise KeyError(f'{identifier.upper()} {name} has no defined in clsName in the config, please specify')
		# Make sure the class exists in the datasets module
		if obj[name].clsName not in available_classes:
			raise NotImplementedError(f'{identifier.upper()} {name} is not available in {identifier}s\n\t'
									  f'in runner.{identifier}.name select one of {available_classes}\n\t'
									  f'or create a dedicated {identifier} class in the {identifier}s module')