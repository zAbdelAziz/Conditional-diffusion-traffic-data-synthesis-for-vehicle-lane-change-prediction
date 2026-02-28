from numpy import ndarray, isin


def split_by_group(p_train: int, p_valid: int, p_test: int, uniq: ndarray, idx: ndarray, vids: ndarray, ignore_test: bool = False):
	if not ignore_test:
		# Normal Train/Valid/Test Splits
		n_train = int(p_train * len(uniq))
		n_valid = int(p_valid * len(uniq))
		# Vehicle Ids
		train_vids = set(uniq[:n_train])
		valid_vids = set(uniq[n_train:n_train + n_valid])
		test_vids = set(uniq[n_train + n_valid:])
	else:
		# if "external" test_dataset provided I split only train/valid from main
		# used in case when I train on Synthetic and test on Original
		n_train = int((p_train / (p_train + p_valid)) * len(uniq))
		# Vehicle Ids
		train_vids = set(uniq[:n_train])
		valid_vids = set(uniq[n_train:])
		test_vids = set()

	train_idx = idx[isin(vids, list(train_vids))]
	valid_idx = idx[isin(vids, list(valid_vids))]
	test_idx = idx[isin(vids, list(test_vids))]

	return train_idx, valid_idx, test_idx, idx

def split_random(p_train: int, p_valid: int, p_test: int, n: int, rng, ignore_test: bool = False):
	if ignore_test:
		# use train/(train+valid) and valid/(train+valid)
		denom = (p_train + p_valid)
		p_train2 = p_train / denom
		perm = rng.permutation(n)
		n_train = int(p_train2 * n)
		train_idx = perm[:n_train]
		valid_idx = perm[n_train:]
		test_idx = None
	else:
		# full train/valid/test from same dataset
		perm = rng.permutation(n)
		n_train = int(p_train * n)
		n_valid = int(p_valid * n)
		train_idx = perm[:n_train]
		valid_idx = perm[n_train:n_train + n_valid]
		test_idx = perm[n_train + n_valid:]
	return train_idx, valid_idx, test_idx