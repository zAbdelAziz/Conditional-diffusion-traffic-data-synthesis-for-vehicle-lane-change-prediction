from trainers.base import BaseTrainer


class SynthTrainer(BaseTrainer):
	def __init__(self, **kwargs):
		super().__init__()

	def start(self):
		pass

	def generate_synthetic(self, **kwargs):
		return [], [], []