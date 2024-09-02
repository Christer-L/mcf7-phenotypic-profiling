import tifffile
import argparse
from tqdm import tqdm
import os
import pickle
from glob import glob

import numpy as np
import SimpleITK as sitk
from skimage import draw
from concurrent.futures import ProcessPoolExecutor
import concurrent


def transform_image(image, mask, image_size=128, circle_radius=50):
    """
    Transforms an input image to align with a predefined circular mask using nonrigid registration.

    Parameters:
    - image: np.array, the input image as a NumPy array.
    - mask: np.array, the mask for the input image as a NumPy array.
    - image_size: int, the size to which the image and mask will be resized.
    - circle_radius: int, the radius of the target circle mask.

    Returns:
    - np.array: Transformed image aligned with the circular mask.
    """
    # Create circle mask for target registration
    black = np.zeros((image_size, image_size))
    rr, cc = draw.disk((image_size / 2, image_size / 2), circle_radius, shape=black.shape)
    black[rr, cc] = 1

    # Initialize registration method
    selx = sitk.ElastixImageFilter()
    selx.SetParameterMap(selx.GetDefaultParameterMap('nonrigid'))
    selx.LogToConsoleOff()

    # Convert NumPy arrays to SimpleITK images
    movingImage = sitk.GetImageFromArray(mask.astype(np.uint8))
    movingLabel = sitk.GetImageFromArray(image)
    fixedImage = sitk.GetImageFromArray(black.astype(np.uint8))

    # Create deformation field necessary to register shape of mask to target circle
    selx.SetMovingImage(movingImage)
    selx.SetFixedImage(fixedImage)
    selx.Execute()

    # Transform image using the deformation field
    resultImage = sitk.Transformix(movingLabel, selx.GetTransformParameterMap())

    # Process outputs
    registeredImage = sitk.GetArrayFromImage(resultImage)
    registeredImage[registeredImage < 0] = 0

    registeredImage = registeredImage / registeredImage.max() * 255.0

    return registeredImage.astype(np.uint8)


def process_image(args):
    rot_img_path, dir_path, idx, gaussian, img_path, seg_path, out_dir = args
    try:
        path_parts = os.path.normpath(dir_path).split(os.sep)
        # ROTATED IMAGE
        rot_img = tifffile.imread(rot_img_path)

        # ROTATED NORMALIZED IMAGE
        rot_norm_img = rot_img / rot_img.max() * 255.0

        # ORIGINAL IMAGE
        img_path = os.path.join(img_path, "original_{}.tif".format(idx))
        img = tifffile.imread(img_path)

        # ROTATED SEGMENTATION
        rot_seg_path = os.path.join(seg_path, "rotated_{}.tif".format(idx))
        rot_seg = tifffile.imread(rot_seg_path)

        # ORIGINAL SEGMENTATION
        seg_path = os.path.join(seg_path, "original_{}.tif".format(idx))
        seg = tifffile.imread(seg_path)

        # Transform images
        gt_img = transform_image(rot_img, rot_seg, image_size=128, circle_radius=50)
        enc1_img = transform_image(img, seg, image_size=128, circle_radius=50)
        enc2_img = rot_norm_img.astype(np.uint8)

        # Save transformed images (gt_img and enc1_img) as needed
        # This part of the code depends on how and where you want to save these images
        out_name = "{}_{}_label{}.tif".format(path_parts[0], path_parts[1], idx)

        enc1_dir = os.path.join(out_dir, "gaussian_{}".format(gaussian), "MEVAEInputs1", "train")
        enc2_dir = os.path.join(out_dir, "gaussian_{}".format(gaussian), "MEVAEInputs2", "train")
        outputs_dir = os.path.join(out_dir, "gaussian_{}".format(gaussian), "MEVAEOutputs", "train")

        os.makedirs(enc1_dir, exist_ok=True)
        os.makedirs(enc2_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)

        tifffile.imwrite(os.path.join(enc2_dir, out_name), enc2_img)
        tifffile.imwrite(os.path.join(enc1_dir, out_name), enc1_img)
        tifffile.imwrite(os.path.join(outputs_dir, out_name), gt_img)

    except Exception as e:
        print(f"Error processing image {rot_img_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="MEVAE data formatting")

    # Add arguments
    parser.add_argument("--pathnames", type=str, help="Pickle file with metadata",
                        nargs='?',
                        default="/mnt/cbib/image_datasets/christer_datasets/mcf7_paper/fixed_paths/batch_0.pkl")
    parser.add_argument("--cells_dir", type=str, help="extracted_cells image dir",
                        nargs='?',
                        default="home/clohk/mcf7_paper/structured/extracted_cells")
    parser.add_argument("--out_dir", type=str, help="output directory for profiles",
                        nargs='?', default="home/clohk/mcf7_paper/structured/MEVAE_data")
    parser.add_argument("--gaussian", type=str, help="Gaussian kernel sigma used in segmentation",
                        nargs='?', default="3")

    args = parser.parse_args()

    paths = []
    for batch in glob(os.path.join(args.pathnames, '*')):
        with open(batch, 'rb') as f:
            batch_paths = pickle.load(f)
            for path in batch_paths:
                paths.append(path)

    # Prepare arguments for each task
    task_args = []
    for dir_path in paths:
        out_dir = args.out_dir
        gaussian = args.gaussian
        cells_dir = args.cells_dir
        img_dir = os.path.join(cells_dir, "images", "gaussian_{}".format(gaussian), dir_path)
        seg_dir = os.path.join(cells_dir, "segmentations", "gaussian_{}".format(gaussian), dir_path)
        rot_img_paths = glob(os.path.join(cells_dir, "images", "gaussian_{}".format(gaussian), dir_path, "rotated_*.tif"))
        for rot_img_path in rot_img_paths:
            idx = rot_img_path.split("_")[-1].split(".")[0]
            task_args.append((rot_img_path, dir_path, idx, gaussian, img_dir, seg_dir, out_dir))
    # Use multiprocessing to process each image
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, arg) for arg in task_args]
        # Optional: use tqdm to show progress
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass


if __name__ == '__main__':
    main()
