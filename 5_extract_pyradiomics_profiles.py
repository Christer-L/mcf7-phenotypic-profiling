import pandas as pd
from radiomics import featureextractor
import argparse
from tqdm import tqdm
import os
import pickle
from glob import glob


def load_pyradiomics_extractor():
    return featureextractor.RadiomicsFeatureExtractor("config/pyradiomics2D.yaml")


def profile(extractor, img_path, label_path, get_cols=True):
    cols = None
    output = extractor.execute(img_path, label_path, label=1)
    values = [float(str(output[k])) for k in output if not k.startswith("diagnostics")]
    if get_cols:
        cols = [k for k in output if not k.startswith("diagnostics")]
    return values, cols


def main():
    parser = argparse.ArgumentParser(description="Pyradiomics profile extraction")

    # Add arguments
    parser.add_argument("--pathnames", type=str, help="Pickle file with metadata",
                        nargs='?',
                        default="/home/christer/Datasets/MCF7/structured/metadata/batches"
                                "/filtered_drugs_dapi_dirs_batch_0.pkl")
    parser.add_argument("--cells_dir", type=str, help="extracted_cells image dir",
                        nargs='?',
                        default="/home/christer/Datasets/MCF7/structured/extracted_cells/"
                        )
    parser.add_argument("--gpu_id", type=str, help="GPU ID", nargs='?', default="7")
    parser.add_argument("--out_dir", type=str, help="output directory for profiles",
                        nargs='?', default="/home/christer/Datasets/MCF7/structured/profiles/pyradiomics")
    parser.add_argument("--gaussian", type=str, help="Gaussian kernel sigma used in segmentation",
                        nargs='?', default="3")

    # Parse arguments
    args = parser.parse_args()

    extractor = load_pyradiomics_extractor()

    with open(args.pathnames, 'rb') as f:
        paths = pickle.load(f)

    for dir_path in tqdm(paths):
        path_parts = os.path.normpath(dir_path).split(os.sep)
        columns = None
        rows = []
        for condition in ["rotated_normalized_*.tif", "rotated_*.tif"]:
            for img_path in glob(os.path.join(args.cells_dir, "images", "gaussian_{}".format(args.gaussian),
                                              dir_path,
                                              "rotated_normalized_*.tif")):
                normalized = condition == "rotated_normalized_*.tif"

                idx = img_path[:-4].split("_")[-1]

                seg_path = os.path.join(args.cells_dir, "segmentations", "gaussian_{}".format(args.gaussian), dir_path,
                                        "rotated_{}.tif".format(idx))

                if columns is None:
                    embedding, cols = profile(extractor, img_path, seg_path, get_cols=True)
                    columns = ["id", "normalized"] + cols
                else:
                    embedding, _ = profile(extractor, img_path, seg_path, get_cols=False)

                entry = [int(idx), normalized] + embedding
                rows.append(entry)

        profile_df = pd.DataFrame(rows, columns=columns)
        save_path = os.path.join(args.out_dir, "gaussian_{}".format(args.gaussian), path_parts[0])
        os.makedirs(save_path, exist_ok=True)
        save_filepath = os.path.join(save_path, "{}.csv".format(path_parts[1]))
        profile_df.to_csv(save_filepath)


if __name__ == '__main__':
    main()
