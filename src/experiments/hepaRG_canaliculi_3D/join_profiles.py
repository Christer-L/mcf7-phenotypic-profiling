import argparse
from glob import glob
import os
import pandas as pd


def main():
    # Arguments
    parser = argparse.ArgumentParser(description="Join profiles from different organoids")

    parser.add_argument("data_dir", type=str,
                        help="Data directory containing all files",
                        nargs="?",
                        default="/mnt/cbib/christers_data/organoids/canaliculi_analysis")

    # Parse arguments
    args = parser.parse_args()

    file_paths = glob(os.path.join(args.data_dir, "*.csv"))
    dfs = [pd.read_csv(path) for path in file_paths]
    df_final = pd.concat(dfs, ignore_index=True)
    df_final.to_csv(os.path.join(args.data_dir, "all_profiles.csv"), index=False)


if __name__ == '__main__':
    main()
