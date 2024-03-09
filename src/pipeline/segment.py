import os
from glob import glob
import tifffile
import numpy as np
from stardist.plot import render_label
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import cv2
import argparse
import pickle


def map_to_16bit(labels):
    unique_values = np.unique(labels)

    # Check if unique values exceed 16-bit range
    if unique_values.size > 65536:
        raise ValueError("Too many unique values to map directly to 16-bit.")

    # Create a mapping from 32-bit to 16-bit values
    mapping = {val: i for i, val in enumerate(unique_values)}

    # Apply mapping to labels
    labels_16bit = np.vectorize(mapping.get)(labels)

    return labels_16bit.astype(np.uint16)


def segment(path_list, gpu_id, kernel_size, data_dir, out_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    with open(path_list, 'rb') as f:
        paths = pickle.load(f)

    # creates a pretrained model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    save_dir = os.path.join(out_dir, "gaussian_{}".format(int(kernel_size)))
    os.makedirs(save_dir, exist_ok=True)

    for path in paths:
        directory = str(path.split('/')[0])
        img_path = os.path.join(data_dir, str(path) + ".tif")
        img = tifffile.imread(img_path)

        # In OpenCV, if you set sigma to 0, it calculates an appropriate sigma based on the kernel size,
        # making the process somewhat adaptive.
        kernel_size = int(kernel_size)
        blurred_image = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        labels, _ = model.predict_instances(normalize(blurred_image))

        # Save segmentation
        labels_16bit = map_to_16bit(labels).astype(np.uint16)

        image_save_dir = os.path.join(save_dir, directory)
        os.makedirs(image_save_dir, exist_ok=True)
        save_path = os.path.join(image_save_dir, os.path.basename(path))
        tifffile.imwrite(save_path, labels_16bit)


def main():
    parser = argparse.ArgumentParser(description="Segmentation")

    # Add arguments
    parser.add_argument("--path_list_file", type=str, help="Path to the pickle file with all DAPI paths",
            nargs='?', default="/home/clohk/datasets/mcf7/metadata/DAPI_paths.pkl")
    parser.add_argument("--gpu_id", type=str, help="GPU ID")
    parser.add_argument("--kernel_size", type=str, help="Gaussian Kernel Size")

    parser.add_argument("--out_dir", type=str, help="Segmentation directory",
            nargs='?', default="/mnt/cbib/image_datasets/christer_datasets/mcf7/segmentations")

    # Parse arguments
    args = parser.parse_args()

    segment(args.path_list_file, args.gpu_id, args.kernel_size, args.data_path, args.out_dir)


if __name__ == '__main__':
    main()
