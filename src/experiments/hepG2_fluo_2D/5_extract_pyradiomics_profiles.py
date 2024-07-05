from concurrent.futures import ProcessPoolExecutor
import concurrent
from glob import glob
import pickle
from tqdm import tqdm
import argparse
import os

from src.pipeline.extract_pyradiomics_profiles import initialize_feature_extractor, extract_features
from src.pipeline.utils import get_all_paths_from_pickle_dir


def main():
    parser = argparse.ArgumentParser(description="DINO profile extraction")

    # Add arguments
    parser.add_argument("--batch_dir", type=str, help="Directory with pickle files",
                        nargs='?',
                        default="/mnt/cbib/christers_data/HepG2/2D_fluo/metadata/paths")
    parser.add_argument("--extracted_cells_dir", type=str, help="extracted_cells dir",
                        nargs='?',
                        default="/mnt/cbib/christers_data/HepG2/2D_fluo/extracted_nuclei")
    parser.add_argument("--out_dir", type=str, help="output directory for profiles",
                        nargs='?', default="/mnt/cbib/christers_data/HepG2/2D_fluo/profiles/pyradiomics")
    parser.add_argument("--gaussian", type=str, help="Gaussian kernel sigma used in segmentation",
                        nargs='?', default="0")

    # Parse arguments
    args = parser.parse_args()

    extractor = initialize_feature_extractor()

    # Get all paths
    paths = [path[:-4] for path in get_all_paths_from_pickle_dir(args.batch_dir)]

    print(len(paths))

    task_args = []
    for dir_path in paths:
        task_args.append((dir_path, extractor, args.gaussian, args.out_dir, args.extracted_cells_dir))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(extract_features, arg) for arg in task_args]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass


if __name__ == '__main__':
    main()
