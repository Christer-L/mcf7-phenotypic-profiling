import argparse

import pandas as pd
import tifffile
import os
from glob import glob
import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor
from tqdm import tqdm
import traceback

from concurrent.futures import ProcessPoolExecutor
import concurrent


def initialize_feature_extractor():
    return featureextractor.RadiomicsFeatureExtractor("src/pipeline/config/pyradiomics3D_shape.yaml")


def profile(pseudo_img, extractor, mask_stack, label_id, columns_exist=True):
    img_itk, mask_itk = numpy_to_itk([pseudo_img, mask_stack])
    output = extractor.execute(img_itk, mask_itk, label=int(label_id))
    values = [
        float(str(output[k]))
        for k in output
        if not k.startswith("diagnostics")
    ]
    row = [int(label_id)] + values
    if not columns_exist:
        columns = ["object_id"] + ["Pyradiomics_{}".format(k) for k in output if not k.startswith("diagnostics")]
        return row, columns
    else:
        return row, None


def numpy_to_itk(images):
    """
    Convert a list of images (or masks) from numpy to SimpleITK format.
    """
    return [sitk.GetImageFromArray(np.squeeze(img)) for img in images]


def get_params_from_path(path):
    path_parts = path.split("/")
    condition = path_parts[-2]
    filename = path_parts[-1]
    name_parts = filename.split("_")
    concentration = name_parts[3]

    if "Control" in name_parts:
        concentration = "0uM"
        condition = "Control"

    return condition, concentration


def extract_features(args):
    try:
        i_path, path, mask, extractor, condition, concentration, min_vol, out_path = args
        pseudo_img = np.zeros_like(mask)
        instance_rows = []
        cols = None

        for mask_id, voxel_count in zip(*np.unique(mask, return_counts=True)):
            if voxel_count < min_vol or mask_id == 0:
                continue
            if cols is None:
                features, cols = profile(pseudo_img, extractor, mask_stack=mask, label_id=mask_id, columns_exist=False)
            else:
                features, _ = profile(pseudo_img, extractor, mask_stack=mask, label_id=mask_id)
            instance_rows.append(features)
        instance_df = pd.DataFrame(instance_rows, columns=cols)
        print(len(instance_df))
        instance_df["condition"] = condition
        instance_df["concentration"] = concentration
        instance_df["path"] = path
        instance_df.to_csv(os.path.join(out_path, "features_{}.csv".format(i_path)), index=False)
    except Exception:
        print(f"Error occurred while processing image: {path}")
        traceback.print_exc()


def main():
    # Arguments
    parser = argparse.ArgumentParser(description="Format image paths")

    parser.add_argument("data_dir", type=str,
                        help="Data directory containing all files",
                        nargs="?",
                        default="/mnt/cbib/image_datasets/christer_datasets/IINS/Ihssane/HepaRG/"
                                "20240326_drug_assay_CPZ_IND/canaliculi_masks")

    parser.add_argument("out_path", type=str,
                        help="Data directory containing all files",
                        nargs="?",
                        default="/mnt/cbib/christers_data/organoids/canaliculi_analysis")

    parser.add_argument("min_volume", type=int,
                        help="Data directory containing all files",
                        nargs="?",
                        default=300)

    # Parse arguments
    args = parser.parse_args()
    paths = glob(os.path.join(args.data_dir, "*", "*"))
    extractor = initialize_feature_extractor()

    task_args = []
    for i_path, path in enumerate(paths):
        condition, concentration = get_params_from_path(path)
        mask = tifffile.imread(path)
        task_args.append((i_path, path, mask, extractor, condition, concentration, args.min_volume, args.out_path))

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(extract_features, arg) for arg in task_args]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass


if __name__ == '__main__':
    main()
