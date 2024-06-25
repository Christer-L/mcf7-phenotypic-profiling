from concurrent.futures import ProcessPoolExecutor
import concurrent
from glob import glob
import pickle
from tqdm import tqdm
import argparse
import os

from src.pipeline.extract_pyradiomics_profiles import initialize_feature_extractor, extract_features


def main():
    parser = argparse.ArgumentParser(description="DINO profile extraction")

    # Add arguments
    parser.add_argument("--batch_dir", type=str, help="Directory with pickle files",
                        nargs='?',
                        default="/mnt/cbib/christers_data/mcf7/structured/metadata/drug_selection/paths")
    parser.add_argument("--extracted_cells_dir", type=str, help="extracted_cells dir",
                        nargs='?',
                        default="/mnt/cbib/christers_data/mcf7/structured/extracted_cells")
    parser.add_argument("--out_dir", type=str, help="output directory for profiles",
                        nargs='?', default="/mnt/cbib/christers_data/mcf7/structured/profiles/pyradiomics")
    parser.add_argument("--gaussian", type=str, help="Gaussian kernel sigma used in segmentation",
                        nargs='?', default="3")

    # Parse arguments
    args = parser.parse_args()

    extractor = initialize_feature_extractor()

    # Get all paths
    paths = []
    for batch in tqdm(glob(os.path.join(args.batch_dir, "*"))):
        with open(batch, 'rb') as f:
            batch_paths = pickle.load(f)
            for path in batch_paths:
                paths.append(path)

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
