
import os
import pickle
import argparse
from src.segmentation.stardist_2D import segment


def main():
    # Arguments
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument("--path_list_file", type=str, help="Path to the pickle file with all paths",
                        nargs='?',
                        default="/mnt/cbib/christers_data/HepG2/2D_fluo/metadata/paths/batch_0.pkl")
    parser.add_argument("--gpu_id", type=str, help="GPU ID", nargs='?', default="0")
    parser.add_argument("--kernel_size", type=str, help="Gaussian Kernel Size", nargs='?', default="0")
    parser.add_argument("--out_dir", type=str, help="Segmentation directory",
                        nargs='?', default="/mnt/cbib/christers_data/HepG2/2D_fluo/segmentations")
    parser.add_argument("--data_root_dir", type=str, help="Data root dir",
                        nargs='?',
                        default="/mnt/cbib/image_datasets/christer_datasets/IINS/Remi/240524-HepG2-Etoposide-2D")

    # Parse arguments
    args = parser.parse_args()

    print("Processing {}".format(args.path_list_file))

    # Select GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Open image paths from the input pickle file
    with open(args.path_list_file, 'rb') as f:
        paths = pickle.load(f)

    segment(paths, args.kernel_size, args.data_root_dir, args.out_dir)


if __name__ == '__main__':
    main()
