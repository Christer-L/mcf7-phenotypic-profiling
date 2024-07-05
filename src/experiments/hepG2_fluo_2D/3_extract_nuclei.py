from tqdm import tqdm
import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor
import argparse
import concurrent
import pickle
from src.drug_selection.format_metadata import save_in_splits
from src.pipeline.extract_image_objects import extract_objects
from src.pipeline.utils import get_max_mask_dim


def main():
    """
    Main function to execute the object extraction process.

    The function performs the following steps:
    1. Parses command line arguments for image dir, segmentations dir, gaussian kernel sigma, and output dir.
    2. Constructs the paths for segmentation files.
    3. Iterates over each segmentation path, creates necessary directories, and appends task arguments for the executor.
    4. Submits the tasks to a ProcessPoolExecutor for concurrent extraction of image objects.

    Note: Break after preparing two tasks for demonstration purposes is commented out.
    """

    # Arguments
    parser = argparse.ArgumentParser(description="Nucleus extraction")

    parser.add_argument("--image_dir", type=str, help="Image directory path (dir that contains plate dirs)",
                        nargs='?',
                        default="/mnt/cbib/image_datasets/christer_datasets/IINS/Remi/240524-HepG2-Etoposide-2D")

    parser.add_argument("--segmentations_dir", type=str, help="Segmentations directory path",
                        nargs='?',
                        default="/mnt/cbib/christers_data/HepG2/2D_fluo/segmentations")

    parser.add_argument("--gaussian", type=str, help="Gaussian kernel sigma used in segmentation",
                        nargs='?',
                        default=0)

    parser.add_argument("--out_dir", type=str, help="output directory",
                        nargs='?',
                        default="/mnt/cbib/christers_data/HepG2/2D_fluo/extracted_nuclei")

    parser.add_argument("--paths_dir", type=str, help="output directory",
                        nargs='?',
                        default="/mnt/cbib/christers_data/HepG2/2D_fluo/metadata/paths")

    parser.add_argument("--n_splits", type=str, help="output directory",
                        nargs='?',
                        default="1")

    # Parse arguments
    args = parser.parse_args()

    task_args = []
    paths = []

    for pickle_file in glob(os.path.join(args.paths_dir, "*.pkl")):
        with open(pickle_file, 'rb') as f:
            paths.extend(pickle.load(f))

    seg_paths = [os.path.join(args.segmentations_dir, "gaussian_{}".format(args.gaussian), path) for path in paths]
    max_dim = get_max_mask_dim(seg_paths)

    # Create directories and append task arguments for executor
    for path in paths:
        # --- Create and record directories ---
        path_parts = path.split(os.sep)

        img_save_dir = os.path.join(args.out_dir, "gaussian_{}".format(args.gaussian),
                                    "images", *path_parts[:-1], path_parts[-1][:-4])
        seg_save_dir = os.path.join(args.out_dir, "gaussian_{}".format(args.gaussian),
                                    "segmentations", *path_parts[:-1], path_parts[-1][:-4])

        os.makedirs(img_save_dir, exist_ok=True)
        os.makedirs(seg_save_dir, exist_ok=True)

        img_path = os.path.join(args.image_dir, path)
        seg_path = os.path.join(args.segmentations_dir, "gaussian_{}".format(args.gaussian), path)
        # -------------------------------------

        task_args.append((img_path, seg_path, img_save_dir, seg_save_dir))

        #For demonstration purposes, break after preparing two tasks
        # if len(task_args) == 5:
        #    break


    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(extract_objects, arg, dim=max_dim + 2, min_n_labels=2) for arg in task_args]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

    filtered_paths = ["/".join(path.split("/")[-4:-1]) for path in glob(os.path.join(args.out_dir,
                                                                                     "gaussian_{}".format(
                                                                                         args.gaussian),
                                                                                     "images",
                                                                                     "*", "*", "*", "*",
                                                                                     "original.tif"))]

    n_removed_images = len(paths) - len(filtered_paths)
    print("Paths removed: ", n_removed_images)

    save_in_splits(filtered_paths, int(args.n_splits), args.metadata_dir)


if __name__ == '__main__':
    main()
