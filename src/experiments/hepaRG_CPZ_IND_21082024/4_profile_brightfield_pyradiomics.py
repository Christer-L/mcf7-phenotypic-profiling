from concurrent.futures import ProcessPoolExecutor
import concurrent
from tqdm import tqdm
import argparse
from glob import glob
import os

from src.pipeline.extract_pyradiomics_profiles import initialize_feature_extractor, extract_features


def main():
    parser = argparse.ArgumentParser(description="DINO profile extraction")

    # Add arguments
    parser.add_argument("--data_dir", type=str, help="Extracted objects dir",
                        nargs='?',
                        default="/mnt/cbib/christers_data/HepaRG/extracted_brightfield_objects/HepaRG_CPZ_IND_21082024")

    parser.add_argument("--out_dir", type=str, help="output directory for profiles",
                        nargs='?', default="/mnt/cbib/christers_data/HepaRG/brightfield_profiles/pyradiomics"
                                           "/HepaRG_CPZ_IND_21082024")

    # Parse arguments
    args = parser.parse_args()
    data_dir = args.data_dir
    out_dir = args.out_dir

    extractor = initialize_feature_extractor()

    # Get all paths
    paths = glob(os.path.join(data_dir, "images", "*", "Fixed", "*", "*_w1Brightfield"))
    formatted_paths = [os.path.join(*path.split(os.sep)[8:]) for path in paths]

    task_args = []
    for dir_path in formatted_paths:
        task_args.append((dir_path, extractor, None, out_dir, data_dir))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(extract_features, arg) for arg in task_args]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass


if __name__ == '__main__':
    main()