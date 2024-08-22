import argparse

import pandas as pd
import tifffile
import os
from glob import glob
import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor
from tqdm import tqdm


def initialize_feature_extractor():
    return featureextractor.RadiomicsFeatureExtractor("src/pipeline/config/pyradiomics3D_shape.yaml")


def profile(extractor, mask_stack, label_id, columns_exist=True):

    # This variable is only needed as an input to the feature extractor.
    # We are only extracting shape features, so the image itself is not used.
    empty_img = np.zeros_like(mask_stack)

    img_itk, mask_itk = numpy_to_itk([empty_img, mask_stack])
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
                        default=50)

    # Parse arguments
    args = parser.parse_args()

    paths = glob(os.path.join(args.data_dir, "*", "*"))

    cols = None
    extractor = initialize_feature_extractor()
    dfs = []

    for path in tqdm(paths[:5]):
        condition, concentration = get_params_from_path(path)
        mask = tifffile.imread(path)

        instance_rows = []

        for mask_id, voxel_count in zip(*np.unique(mask, return_counts=True)):
            if voxel_count < args.min_volume or mask_id == 0:
                continue
            if cols is None:
                features, cols = profile(extractor, mask_stack=mask, label_id=mask_id)
            else:
                features, _ = profile(extractor, mask_stack=mask, label_id=mask_id)

            instance_rows.append(features)
        instance_df = pd.DataFrame(instance_rows, columns=cols)
        instance_df["condition"] = condition
        instance_df["concentration"] = concentration
        instance_df["path"] = path
        dfs.append(instance_df)
    df_all = pd.concat(dfs)
    df_all.to_csv(os.path.join(args.out_path, "features_out.csv"), index=False)


if __name__ == '__main__':
    main()
