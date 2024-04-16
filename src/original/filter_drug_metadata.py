import argparse
import pandas as pd
import os
import pickle
import numpy as np

from collections import OrderedDict

from tqdm import tqdm


def create_batches(paths, num_batches):
    return np.array_split(paths, num_batches)


def save_batch_paths(batch_df, save_path, n_batch):
    paths = []
    path_table = batch_df[["Image_FileName_DAPI", "Image_PathName_DAPI"]]

    for i, row in path_table.iterrows():
        directory = row["Image_PathName_DAPI"].split("/")[1]    # Remove Week# from the beginning of the path
        file_name = row["Image_FileName_DAPI"][:-4]
        dapi_path = os.path.join(directory, file_name)
        print(dapi_path)
        paths.append(dapi_path)

    with open(os.path.join(save_path, 'filtered_drugs_dapi_dirs_batch_{}.pkl'.format(n_batch)), 'wb') as f:
        pickle.dump(paths, f)


def get_high_dose_paths(df_filtered):
    dfs = []
    for drug in list(OrderedDict.fromkeys(df_filtered['Image_Metadata_Compound'])):
        if drug == "DMSO":
            continue
        doses = df_filtered[df_filtered['Image_Metadata_Compound'] == drug]["Image_Metadata_Concentration"].tolist()
        last_doses = list(OrderedDict.fromkeys(doses))[-2:]
        dfs.append(df_filtered[(df_filtered['Image_Metadata_Compound'] == drug) &
                               (df_filtered['Image_Metadata_Concentration'].isin(last_doses))])
        print("hi")
    dose_filtered_df = pd.concat(dfs)

    # TODO: Get only wellplates of dmso that have corresponding drugs
    wellplates = list(OrderedDict.fromkeys(dose_filtered_df["Image_Metadata_Plate_DAPI"]))

    dmso_dose_filtered_df = df_filtered[(df_filtered['Image_Metadata_Compound'] == "DMSO") &
                                        (df_filtered["Image_Metadata_Plate_DAPI"].isin(wellplates))].reset_index(drop=True)
    return pd.concat([dmso_dose_filtered_df, dose_filtered_df])


def filter_drugs(druglist_file, moa_file, metadata_file, save_path):
    drug_list = pd.read_csv(druglist_file, header=None)
    moa_table = pd.read_csv(moa_file).drop(columns=["concentration"])
    metadata_table = pd.read_csv(metadata_file)
    filtered_moa_table = moa_table[moa_table["compound"].isin(drug_list[0].tolist())].drop_duplicates()

    # Save table
    table_path = os.path.join(save_path, "structured", "metadata")
    os.makedirs(table_path, exist_ok=True)
    filtered_moa_table.to_csv(os.path.join(table_path, "moas.csv"), index=False)

    # <<<Get compounds that correspond to each MOA>>>
    # Group compounds based on their MOAs
    grouped = filtered_moa_table.groupby('moa')['compound'].apply(list).to_dict()
    # Determine the maximum list length to standardize the DataFrame shape
    max_length = max(len(compounds) for compounds in grouped.values())

    # Create a new DataFrame with 'moa' as columns and compounds as rows
    # Fill missing values with empty values to ensure equal column lengths
    moas_with_compounds_data = {moa: compounds + [" "] * (max_length - len(compounds))
                                for moa, compounds in grouped.items()}
    moas_with_compounds = pd.DataFrame(moas_with_compounds_data).drop(columns=['DMSO'])

    # Save table
    table_path = os.path.join(save_path, "out", "tables")
    os.makedirs(table_path, exist_ok=True)
    moas_with_compounds.to_csv(os.path.join(table_path, "1_compounds_with_moas.txt"), index=False)

    # <<<Filter unused images from metadata file>>>
    filtered_drugs = filtered_moa_table["compound"].unique()
    filtered_metadata_table = metadata_table[metadata_table["Image_Metadata_Compound"].isin(filtered_drugs)]

    # Save table
    table_path = os.path.join(save_path, "structured", "metadata")
    os.makedirs(table_path, exist_ok=True)
    filtered_metadata_table.to_csv(os.path.join(table_path, "metadata.csv"), index=False)
    filtered_metadata_table = get_high_dose_paths(filtered_metadata_table)

    # Save image batches
    batch_path = os.path.join(save_path, "structured", "metadata", "batches")
    os.makedirs(batch_path, exist_ok=True)
    for i, batch in tqdm(enumerate(create_batches(filtered_metadata_table.reset_index(), 1))):
        save_batch_paths(batch, batch_path, i)


def get_dapi_paths(metadata_file, save_path):
    paths = []
    metadata_table = pd.read_csv(metadata_file)[["Image_FileName_DAPI", "Image_PathName_DAPI"]]
    # TODO: Add progress bar
    for i, row in metadata_table.iterrows():
        directory = row["Image_PathName_DAPI"].split("/")[1]    # Remove Week# from the beginning of the path
        file_name = row["Image_FileName_DAPI"]
        dapi_path = os.path.join(directory, file_name)
        print(dapi_path)
        paths.append(dapi_path)

    # Save filepaths
    list_path = os.path.join(save_path, "structured", "metadata")
    os.makedirs(list_path, exist_ok=True)
    with open(os.path.join(list_path, 'DAPI_paths.pkl'), 'wb') as f:
        pickle.dump(paths, f)


def main():
    parser = argparse.ArgumentParser(description="Data preprocessing")

    # Add arguments
    parser.add_argument("druglist_path", type=str,
                        help=" File path to the list of used drugs (.txt file with drug names separated by newline)")
    parser.add_argument("moa_file", type=int,
                        help=" BBBC021_v1_moa.csv file from https://bbbc.broadinstitute.org/BBBC021")
    parser.add_argument("metadata_file", type=int,
                        help=" BBBC021_v1_image.csv file from https://bbbc.broadinstitute.org/BBBC021")
    parser.add_argument("save_path", type=int,
                        help=" Parent directory for data")

    # Parse arguments
    args = parser.parse_args()
    filter_drugs(args.druglist_path, args.moa_file, args.metadata_file, args.save_path)


if __name__ == '__main__':
    filter_drugs("/home/christer/Datasets/MCF7/metadata/filtered_drugs.txt",
                 "/home/christer/Datasets/MCF7/metadata/BBBC021_v1_moa.csv",
                 "/home/christer/Datasets/MCF7/metadata/BBBC021_v1_image.csv",
                 "/home/christer/Datasets/MCF7/")
    # get_dapi_paths("/home/christer/Datasets/MCF7/metadata/BBBC021_v1_image.csv",
    #                "/home/christer/Datasets/MCF7/")
