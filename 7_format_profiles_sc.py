import os
import pandas as pd
import argparse
from glob import glob
import pickle
import numpy as np
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap
import seaborn as sns

from catboost import Pool, cv
from sklearn.model_selection import StratifiedShuffleSplit
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix


def format_metadata(metadata_path, batch_dir):
    paths_list = []
    for filename in glob(batch_dir + '/*'):
        with open(filename, 'rb') as f:
            dapi_paths = pickle.load(f)
            paths_list.append(dapi_paths)
    paths = [p.split("/")[1] + ".tif" for p in np.concatenate(paths_list)]
    metadata = pd.read_csv(metadata_path)
    filtered_metadata = metadata[metadata["Image_FileName_DAPI"].isin(paths)]
    return filtered_metadata


def main():
    parser = argparse.ArgumentParser(description="Profile formatting")

    # Add arguments
    parser.add_argument("--mevae_dir", type=str, help="Directory containing outputs",
                        nargs='?',
                        default="/home/christer/Datasets/MCF7/structured/MEVAE_data/Output_0-001kl_loss")
    parser.add_argument("--profile_dir", type=str, help="Directory containing profiles",
                        nargs='?',
                        default="/home/christer/Datasets/MCF7/structured/profiles/")
    parser.add_argument("--batch_dir", type=str, help="Directory containing dapi paths (pickle files)",
                        nargs='?',
                        default="/home/christer/Datasets/MCF7/structured/metadata/batches")
    parser.add_argument("--metadata", type=str, help="metadata table",
                        nargs='?',
                        default="/home/christer/Datasets/MCF7/structured/metadata/metadata.csv")
    parser.add_argument("--out_dir", type=str, help="Output directory for profiles",
                        nargs='?', default="/home/christer/Datasets/MCF7/out")
    parser.add_argument("--gaussian", type=str, help="Gaussian kernel sigma used in segmentation",
                        nargs='?', default="3")
    parser.add_argument("--moas", type=str, help="Moa file",
                        nargs='?', default="/home/christer/Datasets/MCF7/structured/metadata/moas.csv")

    args = parser.parse_args()

    metadata = format_metadata(args.metadata, args.batch_dir)
    moa_table = pd.read_csv(args.moas)
    all_profiles = []

    # MEVAE data
    mevae_full_paths = pd.read_csv(os.path.join(args.mevae_dir, "filenames.csv"), header=None)[0].tolist()
    encodings = np.array(pd.read_csv(os.path.join(args.mevae_dir, "encodings.csv"), header=None))

    df_mevae_info = pd.DataFrame()
    df_mevae_info["path"] = mevae_full_paths
    df_mevae_info["encoding_idx"] = range(len(encodings))

    for row_idx, row in tqdm(metadata.iterrows()):
        name = row["Image_FileName_DAPI"][:-4]
        directory = row["Image_PathName_DAPI"].split("/")[1]
        drug = row["Image_Metadata_Compound"]
        moa = moa_table[moa_table["compound"] == drug]["moa"].tolist()[0]

        image_mevae_filtered = df_mevae_info[df_mevae_info["path"].str.contains(name)]
        # Get nucleus label id from path name
        mevae_image_ids = [int(path.split("_")[-1].split(".")[0][5:]) for path in image_mevae_filtered["path"]]
        mevae_embeddings = pd.DataFrame(encodings[image_mevae_filtered["encoding_idx"]],
                                        columns=["MEVAE_{}".format(i) for i in range(64)])
        mevae_embeddings["nucleus_label_id"] = mevae_image_ids

        dino_profile_path = os.path.join(args.profile_dir,
                                         "dino",
                                         "gaussian_{}".format(args.gaussian),
                                         directory,
                                         name + ".csv")
        pyrad_profile_path = os.path.join(args.profile_dir,
                                          "pyradiomics",
                                          "gaussian_{}".format(args.gaussian),
                                          directory,
                                          name + ".csv")

        pyrad_profile = pd.read_csv(pyrad_profile_path)
        pyrad_profile_filtered = pyrad_profile[pyrad_profile["normalized"] == True].drop(
            columns=["Unnamed: 0", "normalized"])
        dino_profile = pd.read_csv(dino_profile_path)
        dino_profile_filtered = dino_profile[dino_profile["normalized"] == True].drop(
            columns=["Unnamed: 0", "normalized"])

        dino_profile_filtered.columns = ["nucleus_label_id"] + ["DINOv2_{}".format(i) for i in range(384)]
        pyrad_profile_filtered.columns = ["nucleus_label_id"] + ["PyRadiomics_{}".format(c) for c in
                                                                 pyrad_profile_filtered.columns
                                                                 if "original" in c]

        img_profiles = pd.concat([
            mevae_embeddings.set_index("nucleus_label_id"),
            dino_profile_filtered.set_index("nucleus_label_id"),
            pyrad_profile_filtered.set_index("nucleus_label_id")
        ], axis=1, join='inner')

        img_profiles["Image_Filename_DAPI"] = row["Image_FileName_DAPI"]
        img_profiles["Image_Metadata_Well_DAPI"] = row["Image_Metadata_Well_DAPI"]
        img_profiles["Image_Metadata_Plate_DAPI"] = row["Image_Metadata_Plate_DAPI"]
        img_profiles["MoA"] = moa
        img_profiles["Image_Metadata_Compound"] = row["Image_Metadata_Compound"]
        all_profiles.append(img_profiles)

    out_df = pd.concat(all_profiles, axis=0)
    out_df.to_csv(os.path.join(args.out_dir, "single_nuclei_gaussian3_normalized_single_cell.csv"))

    return


if __name__ == '__main__':
    main()
