import argparse
import os
from glob import glob
import pandas as pd
from tqdm import tqdm


def read_profiles_from_dir(directory):
    profiles = []
    for wellplate_dir in tqdm(glob(os.path.join(directory, '*'))):
        for profiles_path in glob(os.path.join(wellplate_dir, '*.csv')):
            df_profiles = pd.read_csv(profiles_path, index_col=0)
            df_profiles["Image_PathName_DAPI"] = os.path.basename(profiles_path)
            df_profiles["Image_Metadata_Plate_DAPI"] = os.path.basename(wellplate_dir)
            profiles.append(df_profiles)
        break
    return profiles


def join_profiles(profile_dir, metadata_table, include_mevae=False, gaussian=3):
    base_dir_template = os.path.join(profile_dir, '{}', "gaussian_{}".format(gaussian))
    dino_dir = base_dir_template.format('dino')
    pyradiomics_dir = base_dir_template.format('pyradiomics')
    mevae_dir = base_dir_template.format('mevae') if include_mevae else None

    if mevae_dir:
        raise ValueError('MEVAE profile formatting is not implemented yet')

    metadata_columns = ["Image_FileName_DAPI", "Image_Metadata_Concentration", "Image_Metadata_Compound",
                        "Image_Metadata_Plate_DAPI", "Image_PathName_DAPI"]

    metadata = metadata_table[metadata_columns]
    fixed_wellplate_ids = metadata["Image_PathName_DAPI"].apply(lambda x: x.split("/")[1]).to_list()
    metadata.loc[:, "Image_PathName_DAPI"] = fixed_wellplate_ids

    dino_profiles = read_profiles_from_dir(dino_dir)
    pyradiomics_profiles = read_profiles_from_dir(pyradiomics_dir)
    return


def main():
    parser = argparse.ArgumentParser(description="Gather profiles into one matrix")

    # Add arguments
    parser.add_argument("--profile_dir_path", type=str,
                        help="Path to dir with profile directories",
                        nargs='?', default="/home/christer/Datasets/MCF7_paper/structured/profiles")
    parser.add_argument("--gaussian", type=str,
                        help="Gaussian kernel sigma used in segmentation",
                        nargs='?', default="3")
    parser.add_argument("--filtered_metadata_path", type=str,
                        help="Path to csv file with metadata",
                        nargs='?', default="/home/christer/Datasets/MCF7_paper/metadata/filtered_metadata.csv")
    parser.add_argument('--with_mevae', action=argparse.BooleanOptionalAction, default=False)

    # Parse arguments
    args = parser.parse_args()

    metadata = pd.read_csv(args.filtered_metadata_path, index_col=0)

    join_profiles(args.profile_dir_path, include_mevae=args.with_mevae, gaussian=args.gaussian, metadata_table=metadata)


if __name__ == '__main__':
    main()
