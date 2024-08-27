import argparse
from glob import glob
import os
import pandas as pd


def main():
    # Arguments
    parser = argparse.ArgumentParser(description="Join profiles from different organoids")

    parser.add_argument("pyradiomics_profile_dir", type=str,
                        help="Data directory containing all files",
                        nargs="?",
                        default="/mnt/cbib/christers_data/HepaRG/brightfield_profiles/pyradiomics/"
                                "/HepaRG_CPZ_IND_21082024")

    parser.add_argument("dino_profile_dir", type=str,
                        help="Data directory containing all files",
                        nargs="?",
                        default="/mnt/cbib/christers_data/HepaRG/brightfield_profiles/dino/"
                                "/HepaRG_CPZ_IND_21082024")

    parser.add_argument("out_dir", type=str,
                        help="Data directory containing all files",
                        nargs="?",
                        default="/mnt/cbib/christers_data/HepaRG/brightfield_profiles/all/"
                                "/HepaRG_CPZ_IND_21082024")

    # Parse arguments
    args = parser.parse_args()
    dino_dir = args.dino_profile_dir
    pyrad_dir = args.pyradiomics_profile_dir
    out_dir = args.out_dir

    relative_path = os.path.join("*", "Fixed", "*", "*.csv")

    dino_paths = glob(os.path.join(dino_dir, relative_path))
    pyrad_paths = glob(os.path.join(pyrad_dir, relative_path))

    dino_df = pd.concat([pd.read_csv(path, index_col=0) for path in dino_paths])
    pyrad_df = pd.concat([pd.read_csv(path, index_col=0) for path in pyrad_paths])
    joined_df = pd.merge(pyrad_df, dino_df, on="object_id")

    os.makedirs(out_dir, exist_ok=True)

    joined_df.to_csv(os.path.join(out_dir, "joined_profiles.csv"))


if __name__ == '__main__':
    main()
