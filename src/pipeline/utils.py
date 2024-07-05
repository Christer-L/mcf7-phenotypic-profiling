import os
import numpy as np
import pickle
from skimage.measure import label, regionprops
from tifffile import tifffile
from glob import glob

def save_in_splits(paths, n_splits, path_save_dir):
    splits = np.array_split(paths, int(n_splits))  # Split images into n batches
    # Save filtered metadata table and batches.
    os.makedirs(path_save_dir, exist_ok=True)
    for idx_split, split in enumerate(splits):
        save_path = os.path.join(path_save_dir, "batch_{}.pkl".format(idx_split))
        with open(save_path, 'wb') as f:
            pickle.dump(split, f)


def get_max_mask_dim(segmentations_paths):
    max_dim = 0
    for path in segmentations_paths:
        seg_map = tifffile.imread(path)
        # Get major axis of the mask

        # Label the objects in the mask
        labeled_mask = label(seg_map)

        # Calculate properties of each object
        properties = regionprops(labeled_mask)

        # Extract the length of the major axis for each object
        major_axis_lengths = [prop.major_axis_length for prop in properties if prop.label != 0]

        max_dim = max(max_dim, max(major_axis_lengths))
    return round(max_dim)


def get_all_paths_from_pickle_dir(pickle_dir):
    paths = []
    for pickle_file in glob(os.path.join(pickle_dir, "*.pkl")):
        with open(pickle_file, 'rb') as f:
            paths.extend(pickle.load(f))
    return paths



