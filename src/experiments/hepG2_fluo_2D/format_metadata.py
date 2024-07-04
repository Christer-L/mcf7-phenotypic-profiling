import argparse
from pandas import DataFrame
import pandas as pd
from collections import OrderedDict
import os
import numpy as np
import pickle
from glob import glob


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
    parser = argparse.ArgumentParser(description="Format image paths")

    parser.add_argument("data_dir", type=str,
                        help="Data directory containing all files",
                        nargs="?",
                        default="/mnt/cbib/image_datasets/christer_datasets/")

    parser.add_argument("n_splits", type=int,
                        help="Number of splits of for downstream processing " +
                             "(dependent on how much you can parallelize)",
                        nargs="?",
                        default=5)

    # Parse arguments
    args = parser.parse_args()

    print(glob("/mnt/cbib/image_datasets/christer_datasets/*"))


    # print("Total number of rows (Images - Field of Views): {}".format(len(filtered_table)))
    # print("Total number images corresponding to compounds (excl. DMSO and Taxol): {}".format(n_rows_compounds))
    # print("Total number images corresponding to negative control (DMSO): {}".format(n_rows_neg_ctrl))
    # print("Total number images corresponding to positive control (taxol): {}".format(n_rows_pos_ctrl))
    # print("Total number of compounds: {}".format(len(compounds)))
    # print("Total number of MoAs: {}".format(len(moas)))
    #
    # paths = format_paths(filtered_table)    # Get image paths (each corresponds to a single row in the table)
    # save_in_splits(paths, args.n_splits, args.save_path)  # Save paths in splits
    # filtered_table.to_csv(os.path.join(args.save_path, "filtered_metadata.csv"))


if __name__ == '__main__':
    main()
