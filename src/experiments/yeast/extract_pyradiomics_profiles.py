from src.pipeline.extract_pyradiomics_profiles import initialize_feature_extractor, numpy_to_itk
from concurrent.futures import ProcessPoolExecutor
import concurrent
from glob import glob
import pickle
from tqdm import tqdm
import argparse
import os
import tifffile
import pandas as pd
import traceback

def extract_features(args):
    img_path, seg_path, extractor, save_path = args
    dir_name = os.path.normpath(img_path).split(os.sep)[-2]
    rows = []
    columns = None

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
        os.makedirs(save_path, exist_ok=True)
        save_filepath = os.path.join(save_path, "{}.csv".format(dir_name))

        # Save the DataFrame to a CSV file
        profile_df.to_csv(save_filepath)

    except Exception:
        print(f"Error occurred while processing image: {img_path}")
        print(f"Error occurred while processing segmentation: {seg_path}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="DINO profile extraction")

    parser.add_argument("--extracted_cells_dir", type=str, help="extracted_cells dir",
                        nargs='?',
                        default="/mnt/cbib/christers_data/yeast/extracted_cells")

    parser.add_argument("--out_dir", type=str, help="output directory for profiles",
                        nargs='?', default="/mnt/cbib/christers_data/yeast/profiles/pyradiomics/max_proj")

    # Parse arguments
    args = parser.parse_args()
    extractor = initialize_feature_extractor()

    out_dir = args.out_dir

    paths = glob(os.path.join(args.extracted_cells_dir, "images", "max_proj", "*", "original_normalized.tif"))

    task_args = []
    for img_path in paths:
        seg_path = img_path.replace("original_normalized", "original")
        seg_path = seg_path.replace("/images/", "/masks/")
        task_args.append((img_path, seg_path, extractor, out_dir))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(extract_features, arg) for arg in task_args]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass


if __name__ == '__main__':
    main()
