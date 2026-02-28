from torch.nn import Module


class BaseModel(Module):
	def __init__(self, name: str):
		super().__init__()
		self.name = name
