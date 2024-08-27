import os
from radiomics import featureextractor
import tifffile
import pandas as pd
import numpy as np
import SimpleITK as sitk
import traceback


def initialize_feature_extractor():
    return featureextractor.RadiomicsFeatureExtractor("src/pipeline/config/pyradiomics2D.yaml")


def numpy_to_itk(images):
    """
    Convert a list of images (or masks) from numpy to SimpleITK format.
    """
    return [sitk.GetImageFromArray(np.squeeze(img)) for img in images]


def extract_features(args):
    dir_path, extractor, gaussian, out_dir, extracted_cells_dir = args
    path_parts = os.path.normpath(dir_path).split(os.sep)
    rows = []
    columns = None

    if gaussian:
        img_path = os.path.join(extracted_cells_dir, "gaussian_{}".format(gaussian), "images",
                                dir_path, "original_normalized.tif")
        seg_path = os.path.join(extracted_cells_dir, "gaussian_{}".format(gaussian), "segmentations",
                                dir_path, "original.tif")
    else:
        img_path = os.path.join(extracted_cells_dir, "images",
                                dir_path, "original_normalized.tif")
        seg_path = os.path.join(extracted_cells_dir, "segmentations",
                                dir_path, "original.tif")

    try:
        img_stack = tifffile.imread(img_path)
        mask_stack = tifffile.imread(seg_path)
        n_imgs = img_stack.shape[0]

        for i_z in range(n_imgs):
            img = img_stack[i_z]
            mask = mask_stack[i_z]

            img_itk, mask_itk = numpy_to_itk([img, mask])

            output = extractor.execute(img_itk, mask_itk, label=int(255))
            values = [
                float(str(output[k]))
                for k in output
                if not k.startswith("diagnostics")
            ]
            entry = [int(i_z), str(img_path)] + values
            rows.append(entry)

        if columns is None:
            columns = ["object_id", "path"] + ["Pyradiomics_{}".format(k) for k in output if not k.startswith(
                'diagnostics')]

        # Append the embeddings to a DataFrame
        profile_df = pd.DataFrame(rows, columns=columns)

        # Construct the path to the output CSV file
        if gaussian:
            save_path = os.path.join(out_dir, "gaussian_{}".format(gaussian), os.path.join(*path_parts[:-1]))
        else:
            save_path = os.path.join(out_dir, os.path.join(*path_parts[:-1]))

        os.makedirs(save_path, exist_ok=True)
        save_filepath = os.path.join(save_path, "{}.csv".format(path_parts[-1]))

        # Save the DataFrame to a CSV file
        profile_df.to_csv(save_filepath)
    except Exception:
        print(f"Error occurred while processing image: {img_path}")
        print(f"Error occurred while processing segmentation: {seg_path}")
        traceback.print_exc()
