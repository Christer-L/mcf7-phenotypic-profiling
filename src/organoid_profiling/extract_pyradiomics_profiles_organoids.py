from src.pipeline.extract_pyradiomics_profiles import numpy_to_itk
import tifffile
import pickle
import argparse
from glob import glob
from tqdm import tqdm
import os
import pandas as pd
from radiomics import featureextractor
from concurrent.futures import ProcessPoolExecutor
import concurrent


def extract_features(args):
    dir_path, extractor, out_dir, extracted_cells_dir = args
    path_parts = os.path.normpath(dir_path).split(os.sep)
    rows = []
    columns = None

    img_path = os.path.join(extracted_cells_dir, "images",
                            dir_path, "rotated_normalized.tif")
    seg_path = os.path.join(extracted_cells_dir, "segmentations",
                            dir_path, "rotated.tif")

    img_stack = tifffile.imread(img_path)
    mask_stack = tifffile.imread(seg_path)
    n_imgs = img_stack.shape[0]

    for i_z in range(n_imgs):
        img = img_stack[i_z]
        mask = mask_stack[i_z]

        img_itk, mask_itk = numpy_to_itk([img, mask])

        output = extractor.execute(img_itk, mask_itk, label=int(1))
        values = [
            float(str(output[k]))
            for k in output
            if not k.startswith("diagnostics")
        ]
        print(values)
        entry = [int(i_z)] + values
        rows.append(entry)

    if columns is None:
        columns = ["object_id"] + ["Pyradiomics_{}".format(k) for k in output if not k.startswith("diagnostics")]

    # Append the embeddings to a DataFrame
    profile_df = pd.DataFrame(rows, columns=columns)

    # Construct the path to the output CSV file
    save_path = os.path.join(out_dir, path_parts[0])
    os.makedirs(save_path, exist_ok=True)
    save_filepath = os.path.join(save_path, "{}.csv".format(path_parts[1]))

    # Save the DataFrame to a CSV file
    profile_df.to_csv(save_filepath)


def main():
    parser = argparse.ArgumentParser(description="Pyradiomics profile extraction")

    # Add arguments
    parser.add_argument("--extracted_objects_dir", type=str, help="extracted_objects dir",
                        nargs='?',
                        default="/mnt/cbib/christers_data/organoids/Ihssane_Brightfield_06052024/structured/"
                                "extracted_objects")
    parser.add_argument("--out_dir", type=str, help="output directory for profiles",
                        nargs='?', default="/mnt/cbib/christers_data/organoids/Ihssane_Brightfield_06052024/structured/"
                                           "profiles/pyradiomics")

    # Parse arguments
    args = parser.parse_args()

    extractor = featureextractor.RadiomicsFeatureExtractor()

    # Get all paths
    paths = glob(os.path.join(args.extracted_objects_dir, "images", "*", "*"))
    dirs = ["/".join(path.split("/")[-2:]) for path in paths]

    task_args = []
    for dir_path in dirs:
        task_args.append((dir_path, extractor, args.out_dir, args.extracted_objects_dir))

    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(extract_features, arg) for arg in task_args]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass


if __name__ == '__main__':
    main()
