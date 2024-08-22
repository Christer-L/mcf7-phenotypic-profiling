import argparse
import os
from glob import glob
import tifffile
import numpy as np
from src.pipeline.utils import get_max_mask_dim
import random
from tqdm import tqdm
import concurrent
from concurrent.futures import ProcessPoolExecutor
from src.pipeline.extract_image_objects import extract_objects

def create_striped_array(dimensions, stripe_width):
    height, width = dimensions
    # Create an empty array
    array = np.zeros((height, width))

    # Fill the array with stripes
    for i in range(0, width, stripe_width * 2):
        array[:, i:i + stripe_width] = 1

    return array


def create_striped_array_with_three_values(dimensions, stripe_width, values):
    height, width = dimensions
    # Create an empty array
    array = np.zeros((height, width))

    # Fill the array with stripes of three repeating values
    for i in range(0, width, stripe_width):
        value_index = (i // stripe_width) % 3
        array[:, i:i + stripe_width] = values[value_index]

    return array


def create_checker_array(dimensions, square_size):
    height, width = dimensions
    # Create an empty array
    array = np.zeros((height, width))

    # Fill the array with a checker pattern
    for i in range(0, height, square_size):
        for j in range(0, width, square_size):
            if (i // square_size) % 2 == (j // square_size) % 2:
                array[i:i + square_size, j:j + square_size] = 1

    return array


def create_checker_array_with_three_values(dimensions, square_size, values):
    height, width = dimensions
    # Create an empty array
    array = np.zeros((height, width))

    # Fill the array with a checker pattern with three repeating values
    for i in range(0, height, square_size):
        for j in range(0, width, square_size):
            value_index = ((i // square_size) + (j // square_size)) % 3
            array[i:i + square_size, j:j + square_size] = values[value_index]

    return array


def main():
    # Arguments
    parser = argparse.ArgumentParser(description="Format image paths")

    parser.add_argument("segmentations_dir", type=str,
                        help="Directories with segmentation maps",
                        nargs="?",
                        default="/mnt/cbib/christers_data/mcf7/segmentations/gaussian_3")

    parser.add_argument("out_dir", type=str,
                        help="Output directory",
                        nargs="?",
                        default="/mnt/cbib/christers_data/profiling_test_data/")

    # Parse arguments
    args = parser.parse_args()

    seg_paths = glob(os.path.join(args.segmentations_dir, "*", "*.tif"))
    random.shuffle(seg_paths)
    seg_paths = seg_paths[:601]
    dimensions = tifffile.imread(seg_paths[0]).shape
    # Create background templates for all pattern categories
    patterns = {
        "striped_image_width5": (create_striped_array(dimensions, 5) * 255).astype(np.uint8),
        "striped_image_width2": (create_striped_array(dimensions, 2) * 255).astype(np.uint8),
        "squared_image_width8": (create_checker_array(dimensions, 8) * 255).astype(np.uint8),
        "squared_image_width2": (create_checker_array(dimensions, 2) * 255).astype(np.uint8),
        "squared_three_valued_image": create_checker_array_with_three_values(dimensions, 5,
                                                                             [0, 128, 255]).astype(np.uint8),
        "striped_three_values_image": create_striped_array_with_three_values(dimensions, 3,
                                                                             [0, 128, 255]).astype(np.uint8)
    }

    template_dir = os.path.join(args.out_dir, "templates")
    os.makedirs(template_dir, exist_ok=True)
    for category, pattern in patterns.items():
        tifffile.imwrite(os.path.join(template_dir, category + ".tif"), pattern)

    task_args = []

    max_dim = get_max_mask_dim(seg_paths)

    # Get 100 random paths for each category and generate extracted images
    for category in patterns:
        for i in range(100):
            # Randomly select a segmentation map
            seg_path = seg_paths.pop(0)
            img_path = os.path.join(template_dir, category + ".tif")

            img_save_dir = os.path.join(args.out_dir, "structured", "images", category, str(i))
            seg_save_dir = os.path.join(args.out_dir, "structured", "segmentations", category, str(i))

            os.makedirs(img_save_dir, exist_ok=True)
            os.makedirs(seg_save_dir, exist_ok=True)

            task_args.append((img_path, seg_path, img_save_dir, seg_save_dir))

            # For demonstration purposes, break after preparing tasks
            # if len(task_args) > 5:
            #     break

    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(extract_objects, arg, dim=max_dim + 2, min_n_labels=2) for arg in task_args]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass


            break


    print()


if __name__ == '__main__':
    main()
