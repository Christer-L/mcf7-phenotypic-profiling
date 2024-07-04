import os
import numpy as np
import pickle


def save_in_splits(paths, n_splits, path_save_dir):
    splits = np.array_split(paths, int(n_splits))  # Split images into n batches
    # Save filtered metadata table and batches.
    os.makedirs(path_save_dir, exist_ok=True)
    for idx_split, split in enumerate(splits):
        save_path = os.path.join(path_save_dir, "batch_{}.pkl".format(idx_split))
        with open(save_path, 'wb') as f:
            pickle.dump(split, f)

