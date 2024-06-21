import os
from typing import Tuple

from tqdm import tqdm
from glob import glob
from concurrent.futures import ProcessPoolExecutor
import argparse
import tifffile
import numpy as np
from scipy.ndimage import rotate
import concurrent
from src.drug_selection.format_metadata import save_in_splits


def extract_objects(args, dim=128, normalize_image=False, min_n_labels=5) -> None:
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
        # Load image and segmentation masks
        img = tifffile.imread(img_path)
        segmentations = tifffile.imread(segmentation_path).astype(np.int16)

        # Normalize the image if specified
        img = normalize(img) if normalize_image else img

        # Define and add padding for image and segmentations
        padding_val = dim // 2
        padding = ((padding_val, padding_val), (padding_val, padding_val))
        padded_img = np.pad(img, padding, mode='constant', constant_values=0)
        padded_segmentations = np.pad(segmentations, padding, mode='constant', constant_values=0)

        # Count each label in the mask
        unique_labels = np.unique(segmentations)

        # Confirm that there are at least min_n_labels objects in the image without
        # checking if the object touches the border
        if len(unique_labels) < min_n_labels:
            print(f"Too few objects in image: {img_path}")
            return

        # Create stacks for saving image objects as tif images
        extracted_masks = []
        aligned_masks = []
        extracted_objects = []
        aligned_objects = []
        normalized_extracted_objects = []
        normalized_aligned_objects = []

        for label in unique_labels:
            if label == 0:  # Skip background
                continue

            if object_touches_img_border(label, segmentations):
                continue

            # Create a mask for the single object
            object_mask = (padded_segmentations == label) * 255

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
            mask_to_save = roi_mask.astype(np.uint8)
            rot_mask_to_save = mask_filtered_rotated.astype(np.uint8)

            # Append the processed objects and masks to their respective lists
            extracted_objects.append(obj_to_save)
            aligned_objects.append(rot_obj_to_save)
            normalized_extracted_objects.append(norm_obj_to_save)
            normalized_aligned_objects.append(norm_rot_obj_to_save)
            extracted_masks.append(mask_to_save)
            aligned_masks.append(rot_mask_to_save)

        # Confirm that there are at least min_n_labels objects
        if len(extracted_objects) < min_n_labels:
            print(f"Too few objects in image: {img_path}")
            return

        # Save the object stacks and their masks
        tifffile.imwrite(os.path.join(img_out_dir, "original.tif"), np.array(extracted_objects))
        tifffile.imwrite(os.path.join(img_out_dir, "rotated.tif"), np.array(aligned_objects))
        tifffile.imwrite(os.path.join(img_out_dir, "original_normalized.tif"), np.array(normalized_extracted_objects))
        tifffile.imwrite(os.path.join(img_out_dir, "rotated_normalized.tif"), np.array(normalized_aligned_objects))
        tifffile.imwrite(os.path.join(seg_out_dir, "original.tif"), np.array(extracted_masks))
        tifffile.imwrite(os.path.join(seg_out_dir, "rotated.tif"), np.array(aligned_masks))
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")


def object_touches_img_border(label: int, segmentations: np.ndarray) -> bool:
    """
    Check if object label touches the image border.

    Parameters:
    label (int): The label of the object in the segmentation.
    segmentations (numpy.ndarray): The 2D array representing the segmented image.

    Returns:
    bool: True if the object touches the image border, False otherwise.
    """

    # Find the bounding box (bbox) of the object
    positions = np.where(segmentations == label)
    min_row, max_row = positions[0].min(), positions[0].max()
    min_col, max_col = positions[1].min(), positions[1].max()

    # Return true if object bbox touches the image border
    return (min_row <= 0 or
            min_col <= 0 or
            max_row >= (segmentations.shape[0] - 1) or
            max_col >= (segmentations.shape[1] - 1))


def normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalizes a numpy array (image) to the range [0, 255].

    Parameters:
    img (np.ndarray): The input image as a numpy array.

    Returns:
    np.ndarray: The normalized image as a numpy array.
    """
    return (img - img.min()) / (img.max() - img.min()) * 255


def crop_around_com(img: np.ndarray, mask: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop an image and its mask around the mask's center of mass (COM).

    Parameters:
    img (numpy.ndarray): The 2D array representing the image.
    mask (numpy.ndarray): The 2D array representing the mask.
    dim (int): The dimension of the square crop.

    Returns:
    Tuple[numpy.ndarray, numpy.ndarray]: The cropped mask and image.
    """

    # Calculate the center of mass (COM) of the mask
    y_indices, x_indices = np.nonzero(mask)
    com_row = np.mean(y_indices)
    com_col = np.mean(x_indices)

    # Calculate the top-left corner of the crop based on the COM, ensuring it stays within bounds
    start_row = int(np.clip(com_row - dim / 2, 0, mask.shape[0] - dim))
    start_col = int(np.clip(com_col - dim / 2, 0, mask.shape[1] - dim))

    # Crop the mask and image around the COM
    cropped_mask = mask[start_row:start_row + dim, start_col:start_col + dim]
    cropped_img = img[start_row:start_row + dim, start_col:start_col + dim]

    return cropped_img, cropped_mask


def rotate_centered_object(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate a centered grayscale image object and its mask to vertically align the major axis of the object.

    Parameters:
    image (numpy.ndarray): The 2D array representing the grayscale image object.
    mask (numpy.ndarray): The 2D array representing the mask.

    Returns:
    Tuple[numpy.ndarray, numpy.ndarray]: The rotated image and mask.
    """

    # Compute the orientation of the objects's major axis
    angle = compute_orientation(mask)

    # Calculate the rotation needed to align the major axis vertically.
    # Note: Quality of image after rotation is dependent on order and prefilter args
    rotated_image = rotate(image, angle, reshape=False, order=5, mode='constant', cval=0, prefilter=False)
    rotated_mask = rotate(mask, angle, reshape=False, order=0, mode='constant', cval=0, prefilter=False)

    # Filter out surrounding noise or artefacts from rotation
    rotated_image = rotated_image * rotated_mask

    return rotated_image, rotated_mask


def compute_orientation(mask: np.ndarray) -> float:
    """
    Compute the orientation of the object's major axis.

    Parameters:
    mask (np.ndarray): The 2D array representing the mask of the object.

    Returns:
    float: The angle in degrees that aligns the major axis of the object vertically.

    The function performs the following steps:
    1. Assume mask is a 2D array [H, W].
    2. Calculate centroids of the mask.
    3. Calculate the covariance matrix elements.
    4. Calculate orientation angle in radians, then convert to degrees.
    5. Adjust the angle based on the orientation to ensure vertical alignment.
    """
    # Assume mask is a 2D array [H, W]
    y_coords, x_coords = np.nonzero(mask)
    y_coords, x_coords = y_coords.astype(float), x_coords.astype(float)

    # Calculate centroids
    x_centroid = x_coords.mean()
    y_centroid = y_coords.mean()

    # Calculate the covariance matrix elements
    x_diff = x_coords - x_centroid
    y_diff = y_coords - y_centroid
    mu11 = (x_diff * y_diff).sum()
    mu20 = (x_diff ** 2).sum()
    mu02 = (y_diff ** 2).sum()

    # Calculate orientation angle in radians, then convert to degrees
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    angle = theta * (180 / np.pi)

    # Adjust the angle based on the orientation to ensure vertical alignment
    if angle > 0:
        angle -= 90
    elif angle < 0:
        angle += 90

    return angle


def main():
    """
    Main function to execute the object extraction process.

    The function performs the following steps:
    1. Parses command line arguments for image dir, segmentations dir, gaussian kernel sigma, and output dir.
    2. Constructs the paths for segmentation files.
    3. Iterates over each segmentation path, creates necessary directories, and appends task arguments for the executor.
    4. Submits the tasks to a ProcessPoolExecutor for concurrent extraction of image objects.

    Note: Break after preparing two tasks for demonstration purposes is commented out.
    """

    # Arguments
    parser = argparse.ArgumentParser(description="Image object extraction")
    parser.add_argument("--image_dir", type=str, help="Image directory path (dir that contains plate dirs)",
                        nargs='?',
                        default="/mnt/cbib/christers_data/mcf7/raw_extracted")
    parser.add_argument("--segmentations_dir", type=str, help="Segmentations directory path",
                        nargs='?',
                        default="/mnt/cbib/christers_data/mcf7/segmentations")
    parser.add_argument("--gaussian", type=str, help="Gaussian kernel sigma used in segmentation",
                        nargs='?',
                        default=3)
    parser.add_argument("--out_dir", type=str, help="output directory",
                        nargs='?',
                        default="/mnt/cbib/christers_data/mcf7/structured/extracted_cells")

    parser.add_argument("--metadata_dir", type=str, help="output directory",
                        nargs='?',
                        default="/mnt/cbib/christers_data/mcf7/structured/metadata/drug_selection/filtered_paths")

    parser.add_argument("--n_splits", type=str, help="output directory",
                        nargs='?',
                        default="1")

    # Parse arguments
    args = parser.parse_args()

    task_args = []
    segmentation_paths = os.path.join(args.segmentations_dir, "gaussian_{}".format(args.gaussian), "*", "*")

    # Create directories and append task arguments for executor
    for seg_path in glob(segmentation_paths):

        # --- Create and record directories ---
        path_parts = os.path.normpath(seg_path).split(os.sep)
        gaussian_dir = path_parts[-3]
        plate_dir = path_parts[-2]
        file_name = path_parts[-1]

        img_save_dir = os.path.join(args.out_dir, "images", gaussian_dir, plate_dir, file_name[:-4])
        seg_save_dir = os.path.join(args.out_dir, "segmentations", gaussian_dir, plate_dir, file_name[:-4])

        os.makedirs(img_save_dir, exist_ok=True)
        os.makedirs(seg_save_dir, exist_ok=True)

        img_path = os.path.join(args.image_dir, plate_dir, file_name)
        # -------------------------------------

        task_args.append((img_path, seg_path, img_save_dir, seg_save_dir))

        # For demonstration purposes, break after preparing two tasks
        #if len(task_args) == 2:
        #    break

    filtered_paths = ["/".join(path.split("/")[-3:-1]) for path in glob(os.path.join(args.out_dir,
                                                                                   "images",
                                                                                   "gaussian_{}".format(args.gaussian),
                                                                                   "*",
                                                                                   "*", "original.tif"))]
    save_in_splits(filtered_paths, int(args.n_splits), args.metadata_dir)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(extract_objects, arg) for arg in task_args]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass


if __name__ == '__main__':
    main()
