import argparse
from glob import glob
import os
import tifffile
from src.segmentation.sam import initialize_sam, segment_sam
from src.pipeline.utils import get_largest_cc
import numpy as np
from tqdm import tqdm


def get_relative_path(file_path):
    parts = file_path.split(os.sep)
    relative_path = os.path.join(*parts[7:-1])
    return relative_path


def main():
    # Arguments
    parser = argparse.ArgumentParser(description="Segment brightfield images of organoids")

    parser.add_argument("data_dir", type=str,
                        help="Data directory containing all files",
                        nargs="?",
                        default="/mnt/cbib/image_datasets/christer_datasets/IINS/Shared/HepaRG_CPZ_IND_21082024")

    parser.add_argument("gpu_id", type=int,
                        help="GPU ID",
                        nargs="?",
                        default=7)

    parser.add_argument("out_dir", type=str,
                        help="Output directory",
                        nargs="?",
                        default="/mnt/cbib/christers_data/HepaRG/sam_brightfield_segmentations")

    # Parse arguments
    args = parser.parse_args()
    data_dir = args.data_dir
    gpu_id = args.gpu_id
    out_dir = args.out_dir

    sam = initialize_sam(gpu_id)

    paths = glob(os.path.join(data_dir, "*", "Fixed", "*", "*_w1Brightfield.TIF"))

    for img_path in tqdm(paths):
        save_dir = os.path.join(str(out_dir), str(get_relative_path(img_path)))
        os.makedirs(save_dir, exist_ok=True)

        save_name = os.path.basename(img_path)
        save_path = os.path.join(save_dir, save_name)

        img = tifffile.imread(img_path)
        masks = segment_sam(img, sam)
        filtered_masks = [get_largest_cc(mask_slice) for mask_slice in masks]

        tifffile.imwrite(save_path, filtered_masks)
        print()


if __name__ == '__main__':
    main()
