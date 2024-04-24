import argparse
from pandas import DataFrame
import pandas as pd
from collections import OrderedDict
import os
import numpy as np
import pickle


def filter_compounds(image_metadata_table: DataFrame, compounds) -> DataFrame:
    """
    Get a table from metadata that refers to compounds with defined MoA and their controls from same wellplates.

    :param image_metadata_table: DataFrame containing the image metadata.
    :param compounds: List of compounds to filter.

    :return: DataFrame containing the filtered image metadata.
    """
    selected_image_metadata = []

    metadata_compound = image_metadata_table['Image_Metadata_Compound']
    metadata_wellplate = image_metadata_table['Image_Metadata_Plate_DAPI']

    # Compose dataframe from all rows that correspond to a compound with a defined MoA
    for compound in list(OrderedDict.fromkeys(metadata_compound)):
        if compound not in compounds:
            continue
        selected_image_metadata.append(image_metadata_table[metadata_compound == compound])
    high_conc_image_metadata_df = pd.concat(selected_image_metadata).reset_index(drop=True)

    # Get control rows that originate from the same wellplates as compound images
    compound_wellplates = high_conc_image_metadata_df['Image_Metadata_Plate_DAPI'].tolist()
    control_df = image_metadata_table[(metadata_compound == "DMSO") &
                                      metadata_wellplate.isin(compound_wellplates)].reset_index(drop=True)

    metadata_df = pd.concat([high_conc_image_metadata_df, control_df]).reset_index(drop=True)
    return metadata_df


def format_paths(metadata_table: DataFrame) -> list:
    """
    Return paths from metadata table.

    :param metadata_table: The DataFrame containing metadata information.
    :return: A list of image paths.
    """
    paths = []
    for row_idx, row in metadata_table.iterrows():
        path = os.path.join(row['Image_PathName_DAPI'].split("/")[-1], row['Image_FileName_DAPI'][:-4])
        paths.append(path)
    return paths


def main():
    # Arguments
    parser = argparse.ArgumentParser(description="Format image paths and metadata for drug selection")
    parser.add_argument("moa_metadata_path", type=str,
                        help=" BBBC021_v1_moa.csv file from https://bbbc.broadinstitute.org/BBBC021",
                        nargs="?",
                        default="/mnt/cbib/christers_data/mcf7/metadata/BBBC021_v1_moa.csv")

    parser.add_argument("image_metadata_path", type=str,
                        help=" BBBC021_v1_image.csv file from https://bbbc.broadinstitute.org/BBBC021",
                        nargs="?",
                        default="/mnt/cbib/christers_data/mcf7/metadata/BBBC021_v1_image.csv")

    parser.add_argument("save_path", type=str,
                        help=" Parent directory for formatted metadata",
                        nargs="?",
                        default="/mnt/cbib/christers_data/mcf7/structured/metadata/drug_selection")

    parser.add_argument("n_splits", type=int,
                        help="Number of splits of for downstream processing " +
                             "(dependent on how much you can parallelize)",
                        nargs="?",
                        default=5)

    # Parse arguments
    args = parser.parse_args()
    moa_table = pd.read_csv(args.moa_metadata_path)
    image_metadata_table = pd.read_csv(args.image_metadata_path)

    # Get a list of compounds and MoAs (excluding DMSO - Control).
    compounds = moa_table["compound"].unique().tolist()
    moas = moa_table["moa"].unique().tolist()
    compounds.remove('DMSO')
    moas.remove('DMSO')

    # Filter out compounds (and their controls) that have no defined MoA
    filtered_table = filter_compounds(image_metadata_table, compounds)

    n_rows_compounds = len(filtered_table[
                               (filtered_table["Image_Metadata_Compound"] != "DMSO") &
                               (filtered_table["Image_Metadata_Compound"] != "taxol")])
    n_rows_pos_ctrl = len(filtered_table[filtered_table["Image_Metadata_Compound"] == "taxol"])
    n_rows_neg_ctrl = len(filtered_table[filtered_table["Image_Metadata_Compound"] == "DMSO"])

    print("Total number of rows (Images - Field of Views): {}".format(len(filtered_table)))
    print("Total number images corresponding to compounds (excl. DMSO and Taxol): {}".format(n_rows_compounds))
    print("Total number images corresponding to negative control (DMSO): {}".format(n_rows_neg_ctrl))
    print("Total number images corresponding to positive control (taxol): {}".format(n_rows_pos_ctrl))
    print("Total number of compounds: {}".format(len(compounds)))
    print("Total number of MoAs: {}".format(len(moas)))

    paths = format_paths(filtered_table)    # Get image paths (each corresponds to a single row in the table)
    splits = np.array_split(paths, int(args.n_splits))  # Split images into n batches

    # Save filtered metadata table and batches.
    path_save_dir = os.path.join(args.save_path, "paths")
    os.makedirs(path_save_dir, exist_ok=True)
    for idx_split, split in enumerate(splits):
        save_path = os.path.join(path_save_dir, "batch_{}.pkl".format(idx_split))
        with open(save_path, 'wb') as f:
            pickle.dump(split, f)
    filtered_table.to_csv(os.path.join(args.save_path, "filtered_metadata.csv"))


if __name__ == '__main__':
    main()
