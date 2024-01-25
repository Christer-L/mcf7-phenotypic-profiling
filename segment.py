import os
import toml
from glob import glob
import tifffile
import numpy as np
from stardist.plot import render_label
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import cv2
import argparse


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


def segment(gpu_id, kernel_size):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # creates a pretrained model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    with open('config/config.toml', 'r') as f:
        config = toml.load(f)

    # Access values from the config
    print(config['data']['path'])

    for path_drug in glob(os.path.join(config['data']['path'], "*")):
        for path_dose in glob(os.path.join(path_drug, "*")):
            for path in glob(os.path.join(path_dose, "*/DAPI.tif")):
                print("Segmenting {}".format(path))
                drug = path.split("/")[-4]
                dose = path.split("/")[-3]
                idx = path.split("/")[-2]

                segmentation_dir = os.path.join(config['segmentations']['path'], str(kernel_size),  drug, dose)
                segmentation_path = os.path.join(segmentation_dir, idx + ".tif")

                os.makedirs(segmentation_dir, exist_ok=True)
                img = tifffile.imread(path)

                # In OpenCV, if you set sigma to 0, it calculates an appropriate sigma based on the kernel size,
                # making the process somewhat adaptive.
                blurred_image = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
                labels, _ = model.predict_instances(normalize(blurred_image))

                # Save segmentation
                labels_16bit = map_to_16bit(labels).astype(np.uint16)
                tifffile.imsave(segmentation_path, labels_16bit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segmentation")

    # Add arguments
    parser.add_argument("gpu_id", type=str, help="GPU ID")
    parser.add_argument("kernel_size", type=int, help="Gaussian Kernel Size")

    # Parse arguments
    args = parser.parse_args()
    segment(args.gpu_id, args.kernel_size)





