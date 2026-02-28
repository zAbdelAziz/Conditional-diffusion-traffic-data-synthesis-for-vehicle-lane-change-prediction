import os, random
import numpy as np
from torch import manual_seed, cuda, backends, use_deterministic_algorithms


def set_global_seed(seed: int, deterministic: bool = True):
	os.environ["PYTHONHASHSEED"] = str(seed)

	# Random
	random.seed(seed)
	# Numpy
	np.random.seed(seed)
	# Torch
	manual_seed(seed)
	# Torch Cuda
	cuda.manual_seed_all(seed)

	if deterministic:
		# cuDNN / general determinism
		backends.cudnn.benchmark = False
		backends.cudnn.deterministic = True

		# Avoid TF32 changing math on Ampere+
		backends.cuda.matmul.allow_tf32 = False
		backends.cudnn.allow_tf32 = False

		# Force deterministic kernels where possible
		use_deterministic_algorithms(True)

		# cuBLAS determinism (must be set *before* CUDA context init)
		os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"