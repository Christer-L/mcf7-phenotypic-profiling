import argparse
import os
from glob import glob
from src.pipeline.utils import save_in_splits


def main():
    # Arguments
    parser = argparse.ArgumentParser(description="Format image paths")

    parser.add_argument("data_dir", type=str,
                        help="Data directory containing all files",
                        nargs="?",
                        default="/mnt/cbib/image_datasets/christer_datasets/IINS/Remi/240524-HepG2-Etoposide-2D")

    parser.add_argument("out_path", type=str,
                        help="Data directory containing all files",
                        nargs="?",
                        default="/mnt/cbib/christers_data/HepG2/2D_fluo/metadata/paths")

    parser.add_argument("n_splits", type=int,
                        help="Number of splits of for downstream processing " +
                             "(dependent on how much you can parallelize)",
                        nargs="?",
                        default=1)

    # Parse arguments
    args = parser.parse_args()

    paths = [path for path in glob(os.path.join(args.data_dir, "Batch_*", "40x", "*", "*_w2Epi-405_s*.TIF"))]
    relative_paths = [os.path.sep.join(*path.split(os.path.sep)[-4:]) for path in paths]

    paths_ctrl = [path for path in paths if "/0uM/" in path]
    paths_10uM = [path for path in paths if "/10uM/" in path]
    paths_100uM = [path for path in paths if "/100uM/" in path]

    print("Total number of images (Images - Field of Views): {}".format(len(paths)))
    print("Total number images corresponding to 0uM: {}".format(len(paths_ctrl)))
    print("Total number images corresponding to 10uM: {}".format(len(paths_10uM)))
    print("Total number images corresponding to 100uM: {}".format(len(paths_100uM)))

    save_in_splits(relative_paths, args.n_splits, args.out_path)  # Save paths in pickle file(s)


if __name__ == '__main__':
    main()
