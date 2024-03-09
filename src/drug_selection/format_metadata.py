import argparse
from pandas import DataFrame
import pandas as pd
from collections import OrderedDict
import os
import numpy as np
import pickle


def filter_compounds(image_metadata_table: DataFrame, compounds) -> DataFrame:
    # Filter compounds that have MoA specified and keep only highest concentration
    high_conc_image_metadata = []

    metadata_compound = image_metadata_table['Image_Metadata_Compound']
    metadata_wellplate = image_metadata_table['Image_Metadata_Plate_DAPI']

    for compound in list(OrderedDict.fromkeys(metadata_compound)):
        if compound not in compounds:
            continue
        high_conc_image_metadata.append(image_metadata_table[metadata_compound == compound])

    high_conc_image_metadata_df = pd.concat(high_conc_image_metadata).reset_index(drop=True)
    compound_wellplates = high_conc_image_metadata_df['Image_Metadata_Plate_DAPI'].tolist()
    control_df = image_metadata_table[(metadata_compound == "DMSO") &
                                      metadata_wellplate.isin(compound_wellplates)].reset_index(drop=True)
    metadata_df = pd.concat([high_conc_image_metadata_df, control_df]).reset_index(drop=True)
    return metadata_df


def format_paths(metadata_table: DataFrame) -> DataFrame:
    paths = []
    for row_idx, row in metadata_table.iterrows():
        path = os.path.join(row['Image_PathName_DAPI'].split("/")[-1], row['Image_FileName_DAPI'][:-4])
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Prepare image paths and metadata for drug selection")

    # Add arguments
    parser.add_argument("moa_metadata_path", type=str,
                        help=" BBBC021_v1_moa.csv file from https://bbbc.broadinstitute.org/BBBC021",
                        nargs="?",
                        default="/home/christer/Datasets/MCF7/metadata/BBBC021_v1_moa.csv")

    parser.add_argument("image_metadata_path", type=str,
                        help=" BBBC021_v1_image.csv file from https://bbbc.broadinstitute.org/BBBC021",
                        nargs="?",
                        default="/home/christer/Datasets/MCF7/metadata/BBBC021_v1_image.csv")

    parser.add_argument("save_path", type=str,
                        help=" Parent directory for formatted metadata",
                        nargs="?",
                        default="/home/christer/Datasets/MCF7/structured/metadata/drug_selection/")

    # Parse arguments
    args = parser.parse_args()
    moa_table = pd.read_csv(args.moa_metadata_path)
    image_metadata_table = pd.read_csv(args.image_metadata_path)

    compounds = moa_table["compound"].unique().tolist()
    moas = moa_table["moa"].unique().tolist()
    compounds.remove('DMSO')
    moas.remove('DMSO')

    filtered_table = filter_compounds(image_metadata_table, compounds)

    print("Total number of compounds: {}".format(len(compounds)))
    print("Total number of MoAs: {}".format(len(moas)))

    paths = format_paths(filtered_table)
    splits = np.array_split(paths, 5)

    path_save_dir = os.path.join(args.save_path, "paths")
    os.makedirs(path_save_dir, exist_ok=True)
    for idx_split, split in enumerate(splits):
        save_path = os.path.join(path_save_dir, "batch_{}.pkl".format(idx_split))
        with open(save_path, 'wb') as f:
            pickle.dump(paths, f)

    filtered_table.to_csv(os.path.join(args.save_path, "filtered_metadata.csv"))


if __name__ == '__main__':
    main()
