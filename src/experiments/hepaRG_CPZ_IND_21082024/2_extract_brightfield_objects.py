import argparse
import os
from tqdm import tqdm
from glob import glob
import numpy as np
import tifffile

from src.pipeline.extract_image_objects import (normalize,
                                                object_touches_img_border,
                                                crop_around_com,
                                                rotate_centered_object)

from src.pipeline.utils import get_largest_cc

from concurrent.futures import ProcessPoolExecutor
import concurrent


def extract_objects(args, dim=820) -> None:
    """
    Extracts objects from an image using its corresponding segmentation mask, applies rotation
    and saves them as TIFF stacks.

    Parameters:
    args (tuple): A tuple containing the paths to the image and segmentation mask and both output directories.
    dim (int, optional): The dimension of the square crop around each object's center of mass. Defaults to 128.
    normalize_image (bool, optional): Whether to normalize the image before processing. Defaults to False.

    The function performs the following steps for each object in the image:
    1. Checks if the object touches the image border, and skips it if it does.
    2. Filters out the mask for the single object.
    3. Crops the image and mask around the object's center of mass.
    4. Keeps only the signal under the segmentation mask of the object.
    5. Rotates the object image and mask to align the object's major axis vertically (optional).
    6. Normalizes the signal in the image (optional).
    7. Appends the processed object and mask to their respective lists.

    Finally, the function saves the lists of processed objects and masks as TIFF stacks. Each stack contains images of
    all the objects in the field of view (input image).
    """

    img_path, segmentation_path, img_out_dir, seg_out_dir = args

    try:
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(seg_out_dir, exist_ok=True)

        # Load image and segmentation masks
        img = tifffile.imread(img_path)
        segmentations = tifffile.imread(segmentation_path).astype(np.int16)

        # Define and add padding for image and segmentations
        padding_val = dim // 2
        padding = ((padding_val, padding_val), (padding_val, padding_val))

        # Count each label in the mask
        n_slices = segmentations.shape[0]

        # Create stacks for saving image objects as tif images
        extracted_masks = []
        aligned_masks = []
        extracted_objects = []
        aligned_objects = []
        normalized_extracted_objects = []
        normalized_aligned_objects = []

        for s in range(n_slices):
            if object_touches_img_border(1, segmentations[s]):
                continue

            padded_img = np.pad(img, padding, mode='constant', constant_values=0)

            padded_segmentations = np.pad(segmentations[s], padding, mode='constant', constant_values=0)

            object_mask = get_largest_cc(padded_segmentations)

            # Crop out object with respect to its center of mass (com)
            roi_img, roi_mask = crop_around_com(padded_img, object_mask, dim)

            # Keep only signal under the segmentation mask of the object
            object_filtered = roi_img * roi_mask

            # Rotate object image and mask along its major axis
            object_rotated, mask_filtered_rotated = rotate_centered_object(object_filtered, roi_mask)

            # Normalize signal in the image
            norm_object_unrotated = normalize(object_filtered)
            norm_object_rotated = normalize(object_rotated)

            # Convert the images and masks to the appropriate data types for saving
            obj_to_save = object_filtered.astype(np.uint32)
            rot_obj_to_save = object_rotated.astype(np.uint32)
            norm_obj_to_save = norm_object_unrotated.astype(np.uint8)
            norm_rot_obj_to_save = norm_object_rotated.astype(np.uint8)
            mask_to_save = (roi_mask * 255).astype(np.uint8)
            rot_mask_to_save = (mask_filtered_rotated * 255).astype(np.uint8)

            if (object_touches_img_border(255, mask_to_save) or
                    object_touches_img_border(255, rot_mask_to_save)):
                continue

            # Append the processed objects and masks to their respective lists
            extracted_objects.append(obj_to_save)
            aligned_objects.append(rot_obj_to_save)
            normalized_extracted_objects.append(norm_obj_to_save)
            normalized_aligned_objects.append(norm_rot_obj_to_save)
            extracted_masks.append(mask_to_save)
            aligned_masks.append(rot_mask_to_save)

        # Save the object stacks and their masks
        tifffile.imwrite(os.path.join(img_out_dir, "original.tif"), np.array(extracted_objects))
        tifffile.imwrite(os.path.join(img_out_dir, "rotated.tif"), np.array(aligned_objects))
        tifffile.imwrite(os.path.join(img_out_dir, "original_normalized.tif"), np.array(normalized_extracted_objects))
        tifffile.imwrite(os.path.join(img_out_dir, "rotated_normalized.tif"), np.array(normalized_aligned_objects))
        tifffile.imwrite(os.path.join(seg_out_dir, "original.tif"), np.array(extracted_masks))
        tifffile.imwrite(os.path.join(seg_out_dir, "rotated.tif"), np.array(aligned_masks))
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Object extraction from brightfield images")
    parser.add_argument("data_dir", type=str,
                        help="Data directory containing all files",
                        nargs="?",
                        default="/mnt/cbib/image_datasets/christer_datasets/IINS/Shared/HepaRG_CPZ_IND_21082024")

    parser.add_argument("--segmentations_dir", type=str, help="Segmentations directory path",
                        nargs='?',
                        default="/mnt/cbib/christers_data/HepaRG/sam_brightfield_segmentations/HepaRG_CPZ_IND_21082024")

    parser.add_argument("--out_dir", type=str, help="output directory",
                        nargs='?',
                        default="/mnt/cbib/christers_data/HepaRG/extracted_brightfield_objects/HepaRG_CPZ_IND_21082024")

    # Parse arguments
    args = parser.parse_args()
    data_dir = args.data_dir
    segmentations_dir = args.segmentations_dir
    out_dir = args.out_dir

    task_args = []

    segmentation_paths = glob(os.path.join(segmentations_dir, "*", "Fixed", "*", "*.TIF"))

    # Create directories and append task arguments for executor
    for seg_path in tqdm(segmentation_paths, total=len(segmentation_paths)):
        # --- Create and record directories ---
        img_relative_path_parts = seg_path.split(os.sep)[-4:]
        img_path = os.path.join(data_dir, *img_relative_path_parts)
        img_save_dir = os.path.join(out_dir, "images", *img_relative_path_parts)[:-4]
        seg_save_dir = os.path.join(out_dir, "segmentations", *img_relative_path_parts)[:-4]
        task_args.append((img_path, seg_path, img_save_dir, seg_save_dir))

    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(extract_objects, arg) for arg in task_args]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass


if __name__ == '__main__':
    main()
