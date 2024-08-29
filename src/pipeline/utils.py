import os
from os import major

import numpy as np
import pickle
from skimage.measure import label, regionprops
from tifffile import tifffile
from glob import glob
from tqdm import tqdm
import cv2

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
    for path in tqdm(segmentations_paths):
        seg_map = tifffile.imread(path)

        # Extract the length of the major axis for each object
        major_axis_length = get_mask_dim(seg_map)

        max_dim = max(max_dim, major_axis_length)
    return round(max_dim)


def get_mask_dim(segmentation, label_val=1):
    labeled_mask = label(segmentation)

    # Calculate properties of each object
    properties = regionprops(labeled_mask)

    if len(properties) == 0:
        return 0

    axis_length = [prop.major_axis_length for prop in properties if prop.label == label_val]

    if len(axis_length) == 0:
        return 0

    # Extract the length of the major axis for each object
    major_axis_length = max(axis_length)

    return major_axis_length


def get_max_mask_dim_stacks(segmentations_paths):
    max_dim = 0
    print("Scanning through segmentations...")
    for path in tqdm(segmentations_paths):
        seg_map = tifffile.imread(path)

        for stack_slice in seg_map:
            # Label the objects in the mask
            labeled_mask = label(stack_slice)

            # Calculate properties of each object
            properties = regionprops(labeled_mask)

            if len(properties) == 0:
                continue

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


def get_largest_cc(segmentation):
    labels = label(segmentation)
    largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largest_cc


def remove_boundary_touching_slices(seg_stack):
    # Initialize a list to store the filtered slices
    filtered_slices = []

    for i, seg in enumerate(seg_stack):
        # Find contours in the segmentation mask
        contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Flag to determine if any contour touches the boundary
        touches_boundary = False

        for cnt in contours:
            if any([pt[0][0] == 0 or pt[0][0] == seg.shape[1] - 1 or pt[0][1] == 0 or pt[0][1] == seg.shape[0] - 1 for
                    pt in cnt]):
                touches_boundary = True
                break

        # If no object touches the boundary, add the slice to the filtered list
        if not touches_boundary:
            filtered_slices.append(seg)

    # Convert the filtered list back to a numpy array
    return np.array(filtered_slices)


def filter_masks(seg_stack, min_len=40, max_len=800):
    # Initialize a list to store the filtered slices
    filtered_slices = []

    for i, seg in enumerate(seg_stack):
        if get_mask_dim(seg) >= min_len and get_mask_dim(seg) <= max_len:
            filtered_slices.append(seg)

    # Convert the filtered list back to a numpy array
    return np.array(filtered_slices)
