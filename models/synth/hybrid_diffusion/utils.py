from torch import Tensor, empty

def expand_time_indexed(buf_1d: Tensor, t: Tensor, ref: Tensor) -> Tensor:
	"""
	Expand a [T] diffusion buffer indexed by t:[B] to match ref rank.

	Example:
		buf_1d: [T]
		t:      [B]
		ref:    [B, T_seq, D]

	returns:
		[B, 1, 1]
	"""
	out = buf_1d[t]
	view_shape = [t.shape[0]] + [1] * (ref.dim() - 1)
	return out.view(*view_shape)


def empty_like_1d(x: Tensor) -> Tensor:
	return empty(x.shape, device=x.device, dtype=x.dtype)