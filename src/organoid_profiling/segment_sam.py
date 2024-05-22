import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import argparse
from ultralytics import SAM
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

from pathlib import Path
import tifffile


def segment_sam(img, model_sam):
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    results_sam = model_sam(img)
    masks = np.array(results_sam[0].masks.data.cpu() * 255, dtype=np.uint8)
    return masks


def main():
    # Arguments
    parser = argparse.ArgumentParser(description="Organoid segmentation")
    parser.add_argument("--image_dir", type=str, help="Image directory path (stacks of raw data)",
                        nargs='?',
                        default="/mnt/cbib/christers_data/organoids/Ihssane_Brightfield_06052024/raw")
    parser.add_argument("--segmentations_dir", type=str, help="Segmentations directory path",
                        nargs='?',
                        default="/mnt/cbib/christers_data/organoids/Ihssane_Brightfield_06052024/structured"
                                "/segmentations")

    # Parse arguments
    args = parser.parse_args()
    input_dir = args.image_dir
    output_dir = args.segmentations_dir

    model_sam = SAM('models/sam_b.pt').cuda()
    img_paths = glob(os.path.join(input_dir, "*"))
    for img_path in img_paths:
        img_name = Path(img_path).stem
        img_stack = tifffile.imread(img_path)
        for i_z in tqdm(range(img_stack.shape[0])):
            img = img_stack[i_z]
            save_dir = os.path.join(output_dir, img_name)
            os.makedirs(save_dir, exist_ok=True)
            save_name = os.path.join(save_dir, str(i_z) + ".tif")
            masks = segment_sam(img, model_sam)
            tifffile.imwrite(save_name, masks)


if __name__ == '__main__':
    main()
