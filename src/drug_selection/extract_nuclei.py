from tqdm import tqdm
import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor
import argparse
import concurrent
from format_metadata import save_in_splits
from src.pipeline.extract_image_objects import extract_objects


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
    parser = argparse.ArgumentParser(description="Image object extraction")

    parser.add_argument("--image_dir", type=str, help="Image directory path (dir that contains plate dirs)",
                        nargs='?',
                        default="/mnt/cbib/christers_data/mcf7/raw_extracted")

    parser.add_argument("--segmentations_dir", type=str, help="Segmentations directory path",
                        nargs='?',
                        default="/mnt/cbib/christers_data/mcf7/segmentations")

    parser.add_argument("--gaussian", type=str, help="Gaussian kernel sigma used in segmentation",
                        nargs='?',
                        default=3)

    parser.add_argument("--out_dir", type=str, help="output directory",
                        nargs='?',
                        default="/mnt/cbib/christers_data/mcf7/structured/extracted_cells")

    parser.add_argument("--metadata_dir", type=str, help="output directory",
                        nargs='?',
                        default="/mnt/cbib/christers_data/mcf7/structured/metadata/drug_selection/filtered_paths")

    parser.add_argument("--n_splits", type=str, help="output directory",
                        nargs='?',
                        default="1")

    # Parse arguments
    args = parser.parse_args()

    task_args = []
    segmentation_paths = os.path.join(args.segmentations_dir, "gaussian_{}".format(args.gaussian), "*", "*")

    # Create directories and append task arguments for executor
    for seg_path in glob(segmentation_paths):
        # --- Create and record directories ---
        path_parts = os.path.normpath(seg_path).split(os.sep)
        gaussian_dir = path_parts[-3]
        plate_dir = path_parts[-2]
        file_name = path_parts[-1]

        img_save_dir = os.path.join(args.out_dir, "images", gaussian_dir, plate_dir, file_name[:-4])
        seg_save_dir = os.path.join(args.out_dir, "segmentations", gaussian_dir, plate_dir, file_name[:-4])

        os.makedirs(img_save_dir, exist_ok=True)
        os.makedirs(seg_save_dir, exist_ok=True)

        img_path = os.path.join(args.image_dir, plate_dir, file_name)
        # -------------------------------------

        task_args.append((img_path, seg_path, img_save_dir, seg_save_dir))

        # For demonstration purposes, break after preparing two tasks
        # if len(task_args) == 2:
        #    break

    filtered_paths = ["/".join(path.split("/")[-3:-1]) for path in glob(os.path.join(args.out_dir,
                                                                                     "images",
                                                                                     "gaussian_{}".format(
                                                                                         args.gaussian),
                                                                                     "*",
                                                                                     "*", "original.tif"))]
    save_in_splits(filtered_paths, int(args.n_splits), args.metadata_dir)

    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(extract_objects, arg) for arg in task_args]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass


if __name__ == '__main__':
    main()
