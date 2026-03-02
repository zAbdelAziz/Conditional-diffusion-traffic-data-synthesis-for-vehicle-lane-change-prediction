from pathlib import Path
from typing import Any, Dict

from torch.nn import Module
from torchview import draw_graph


def model_params(model: Module, out_path: Path):
	"""Always-available text summary: repr + param counts."""
	total = sum(p.numel() for p in model.parameters())
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	text = f'Model summary:\n\tTotal Paramters:\t{total}\n\tTrainable Paramters:\t{trainable}'
	with out_path.open("w", encoding="utf-8") as f:
		f.write(text)
	return text

def plot_model(model: Module, sample_inputs: Any, out_path: Path):
	g = draw_graph(model, input_data=sample_inputs, expand_nested=True, device=next(model.parameters()).device)
	g.visual_graph.render(str(out_path.with_suffix("")), format="png")
	return out_path
