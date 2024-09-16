from concurrent.futures import ProcessPoolExecutor
import concurrent
from tqdm import tqdm
import argparse
from glob import glob
import os
import tifffile
import traceback
import pandas as pd
import SimpleITK as sitk
import numpy as np

from src.pipeline.extract_pyradiomics_profiles import  numpy_to_itk
from radiomics import featureextractor


def initialize_feature_extractor_3D():
    return featureextractor.RadiomicsFeatureExtractor("src/pipeline/config/pyradiomics3D.yaml")


def extract_features(args):
    img_path, mask_path, compound, dose, extractor, save_path = args

    try:
        rows = []
        img = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)

        if np.unique(mask).size == 1:
            print(img_path, " had an empy mask!")
            traceback.print_exc()
            return

        img_itk, mask_itk = numpy_to_itk([img, mask])

        output = extractor.execute(img_itk, mask_itk, label=int(255))
        values = [
            float(str(output[k]))
            for k in output
            if not k.startswith("diagnostics")
        ]
        entry = [str(img_path), compound, dose] + values
        rows.append(entry)

        columns = ["Path", 'Compound', 'Dose'] + ["Pyradiomics_{}".format(k) for k in output if not k.startswith(
            'diagnostics')]

        # Append the embeddings to a DataFrame
        profile_df = pd.DataFrame(rows, columns=columns)

        # Save the DataFrame to a CSV file
        profile_df.to_csv(save_path)
    except Exception:
        print(f"Error occurred while processing image: {img_path}")
        print(f"Error occurred while processing segmentation: {mask_path}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="DINO profile extraction")

    # Add arguments
    parser.add_argument("--actin_mask_dir", type=str, help="Organoid segmentation mask dir",
                        nargs='?',
                        default="/mnt/cbib/christers_data/HepaRG/actin_segmentations")

    parser.add_argument("--img_dir", type=str, help="Image dir",
                        nargs='?',
                        default="/mnt/cbib/image_datasets/christer_datasets/IINS/Shared/HepaRG_CPZ_IND_21082024")

    parser.add_argument("--out_dir", type=str, help="Save dir",
                        nargs='?',
                        default="/mnt/cbib/christers_data/HepaRG/nucleus_profiles")

    # Parse arguments
    args = parser.parse_args()
    actin_mask_dir = args.actin_mask_dir
    img_dir = args.img_dir
    out_dir = args.out_dir

    extractor = initialize_feature_extractor_3D()

    # Get all paths
    actin_mask_paths = glob(os.path.join(actin_mask_dir, "*", "Fixed", "*", "*.TIF"))

    task_args = []
    for i, actin_mask_path in enumerate(actin_mask_paths):
        relative_path = os.path.relpath(actin_mask_path, actin_mask_dir).split("w4soSPIM-")[0]
        img_path = os.path.join(img_dir, relative_path + "w2soSPIM-405-Stream.TIF")

        condition_info = img_path.split(os.sep)[-2]
        condition_parts = condition_info.split("_")
        compound = condition_parts[2]
        dose = condition_parts[3]

        save_path = os.path.join(out_dir, f"df_{i}.csv")

        task_args.append((img_path, actin_mask_path, compound, dose, extractor, save_path))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(extract_features, arg) for arg in task_args]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass


if __name__ == '__main__':
    main()