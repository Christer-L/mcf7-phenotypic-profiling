import argparse
import os
import pickle
from tqdm import tqdm
from radiomics import featureextractor
from glob import glob
import tifffile
import pandas as pd
import numpy as np
import SimpleITK as sitk


from concurrent.futures import ProcessPoolExecutor
import concurrent


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

    img_path = os.path.join(extracted_cells_dir, "images", "gaussian_{}".format(gaussian),
                            dir_path, "rotated_normalized.tif")
    seg_path = os.path.join(extracted_cells_dir, "segmentations", "gaussian_{}".format(gaussian),
                            dir_path, "rotated.tif")

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
        entry = [int(i_z)] + values
        rows.append(entry)

    if columns is None:
        columns = ["object_id"] + ["Pyradiomics_{}".format(k) for k in output if not k.startswith("diagnostics")]

    # Append the embeddings to a DataFrame
    profile_df = pd.DataFrame(rows, columns=columns)

    # Construct the path to the output CSV file
    save_path = os.path.join(out_dir, "gaussian_{}".format(gaussian), path_parts[0])
    os.makedirs(save_path, exist_ok=True)
    save_filepath = os.path.join(save_path, "{}.csv".format(path_parts[1]))

    # Save the DataFrame to a CSV file
    profile_df.to_csv(save_filepath)


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

    extractor = featureextractor.RadiomicsFeatureExtractor("../src/pipeline/config/pyradiomics2D.yaml")

    # Get all paths
    paths = []
    for batch in tqdm(glob(os.path.join(args.batch_dir, "*"))):
        with open(batch, 'rb') as f:
            batch_paths = pickle.load(f)
            for path in batch_paths:
                print(path)
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
