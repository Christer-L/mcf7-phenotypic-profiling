import argparse
import os
import pickle
from tqdm import tqdm
from radiomics import featureextractor
from glob import glob
import tifffile
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
import concurrent


def extract_features(args):
    dir_path, extractor, gaussian, out_dir, extracted_cells_dir = args
    path_parts = os.path.normpath(dir_path).split(os.sep)
    rows = []

    for img_path in glob(os.path.join(extracted_cells_dir, "images", "gaussian_{}".format(gaussian),
                                      dir_path, "rotated_normalized_*.tif")):
        idx = img_path[:-4].split("_")[-1]

        seg_path = os.path.join(extracted_cells_dir, "segmentations", "gaussian_{}".format(gaussian),
                                dir_path, "rotated_{}.tif".format(idx))

        output = extractor.execute(img_path, seg_path, label=int(1))
        values = [
            float(str(output[k]))
            for k in output
            if not k.startswith("diagnostics")
        ]
        entry = [int(idx)] + values
        rows.append(entry)

    columns = ["index"] + ["Pyradiomics_{}".format(k) for k in output if not k.startswith("diagnostics")]
    profile_df = pd.DataFrame(rows, columns=columns)
    save_path = os.path.join(out_dir, "gaussian_{}".format(gaussian), path_parts[0])
    os.makedirs(save_path, exist_ok=True)
    save_filepath = os.path.join(save_path, "{}.csv".format(path_parts[1]))
    profile_df.to_csv(save_filepath)


def main():
    parser = argparse.ArgumentParser(description="DINO profile extraction")

    # Add arguments
    parser.add_argument("--batch_dir", type=str, help="Directory with pickle files",
                        nargs='?',
                        default="/mnt/cbib/image_datasets/christer_datasets/mcf7_paper/fixed_paths")
    parser.add_argument("--extracted_cells_dir", type=str, help="extracted_cells dir",
                        nargs='?',
                        default="/home/clohk/mcf7_paper/structured/extracted_cells")
    parser.add_argument("--out_dir", type=str, help="output directory for profiles",
                        nargs='?', default="/home/clohk/mcf7_paper/structured/profiles/pyradiomics")
    parser.add_argument("--gaussian", type=str, help="Gaussian kernel sigma used in segmentation",
                        nargs='?', default="3")

    # Parse arguments
    args = parser.parse_args()

    extractor = featureextractor.RadiomicsFeatureExtractor()

    paths = []
    for batch in tqdm(glob(os.path.join(args.batch_dir, "*"))):
        with open(batch, 'rb') as f:
            batch_paths = pickle.load(f)
            for path in batch_paths:
                paths.append(path)

    task_args = []
    for dir_path in paths:
        task_args.append((dir_path, extractor, args.gaussian, args.out_dir, args.extracted_cells_dir))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(extract_features, arg) for arg in task_args]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass


if __name__ == '__main__':
    main()
