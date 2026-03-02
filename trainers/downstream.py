from trainers.base import BaseTrainer


class DownstreamTrainer(BaseTrainer):
	def __init__(self, train_dataset, model, test_dataset=None):
		super().__init__(name="downstream", train_dataset=train_dataset, test_dataset=test_dataset, model=model)

	def start(self):
		pass