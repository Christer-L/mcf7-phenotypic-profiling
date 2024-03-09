import os
from tqdm import tqdm
from glob import glob
from concurrent.futures import ProcessPoolExecutor
import argparse
import tifffile
import numpy as np
from scipy.ndimage import rotate
import concurrent


def extract_nuclei(args, dim=128, normalize_image=False):
    img_path, segmentation_path, img_out_dir, seg_out_dir = args
    try:
        img_np = tifffile.imread(img_path)

        if normalize_image:
            img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255)

        mask_np = tifffile.imread(segmentation_path)

        # Define padding for later processing
        padding_val = dim // 2
        padding = ((padding_val, padding_val), (padding_val, padding_val))

        # Load images and masks into PyTorch tensors
        img = img_np.astype(np.uint8)
        mask = mask_np.astype(np.int16)

        padded_img = np.pad(img_np, padding, mode='constant', constant_values=0)
        padded_mask = np.pad(mask_np, padding, mode='constant', constant_values=0)

        # Process each label in the mask
        unique_labels = np.unique(mask)
        for label in unique_labels:
            if label == 0:  # Skip background
                continue

            # Find the bounding box of the current nucleus
            positions = np.where(mask == label)
            min_row, max_row = positions[0].min(), positions[0].max()
            min_col, max_col = positions[1].min(), positions[1].max()

            if min_row <= 0 or min_col <= 0 or max_row >= (img.shape[0] - 1) or max_col >= (img.shape[1] - 1):
                continue

            # Create a mask for the current nucleus
            nucleus_mask = (padded_mask == label)

            roi_mask, roi_img = center_crop_around_com(padded_img, nucleus_mask, dim)

            nucleus_filtered = roi_img * roi_mask

            nucleus_filtered_rotated, mask_filtered_rotated = rotate_centered_cell(nucleus_filtered, roi_mask)

            norm_nucleus_filtered = normalize_single_nuc(nucleus_filtered)
            norm_nucleus_filtered_rotated = normalize_single_nuc(nucleus_filtered_rotated)

            mask_to_save = roi_mask.astype(np.uint8)
            rot_mask_to_save = mask_filtered_rotated.astype(np.uint8)
            norm_rot_nuc_to_save = norm_nucleus_filtered_rotated.astype(np.uint8)
            norm_nuc_to_save = norm_nucleus_filtered.astype(np.uint8)

            # Save the centered nucleus and its mask
            name = f"{label.item()}.tif"
            tifffile.imwrite(os.path.join(img_out_dir, "original_{}".format(name)),
                             nucleus_filtered)
            tifffile.imwrite(os.path.join(seg_out_dir, "original_{}".format(name)),
                             mask_to_save)
            tifffile.imwrite(os.path.join(img_out_dir, "rotated_{}".format(name)),
                             nucleus_filtered_rotated)
            tifffile.imwrite(os.path.join(seg_out_dir, "rotated_{}".format(name)),
                             rot_mask_to_save)
            tifffile.imwrite(os.path.join(img_out_dir, "rotated_normalized_{}".format(name)),
                             norm_rot_nuc_to_save)
            tifffile.imwrite(os.path.join(img_out_dir, "original_normalized_{}".format(name)),
                             norm_nuc_to_save)
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")


def normalize_single_nuc(img):
    return (img - img.min()) / (img.max() - img.min()) * 255


def center_crop_around_com(img, mask, dim):
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

    return cropped_mask, cropped_img


def rotate_centered_cell(image, mask):
    """Rotate a centered grayscale cell image and its mask to align the major axis vertically."""
    angle = compute_orientation(mask)

    # Calculate the center of the image for rotation
    center = (image.shape[1] / 2 - 0.5, image.shape[0] / 2 - 0.5)

    # Calculate the rotation needed to align the major axis vertically.
    rotated_image = rotate(image, angle, reshape=False, order=1, mode='constant', cval=0, prefilter=True)
    rotated_mask = rotate(mask, angle, reshape=False, order=0, mode='constant', cval=0, prefilter=False)

    return rotated_image, rotated_mask


def compute_orientation(mask):
    """Compute the orientation of the cell's major axis using NumPy."""
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
    parser = argparse.ArgumentParser(description="Cell extraction")

    # Add arguments
    parser.add_argument("--image_dir", type=str, help="Image directory path (dir that contains plate dirs)",
                        nargs='?', default="/mnt/cbib/image_datasets/christer_datasets/archive2024/mcf7/raw_extracted")
    parser.add_argument("--segmentations_dir", type=str, help="Segmentations directory path",
                        nargs='?', default="/mnt/cbib/christers_data/mcf7/segmentations/gaussian_3")
    parser.add_argument("--out_dir", type=str, help="output directory",
                        nargs='?', default="/home/clohk/mcf7_paper/structured/extracted_cells")

    # Parse arguments
    args = parser.parse_args()
    task_args = []

    for seg_path in glob(os.path.join(args.segmentations_dir, "*", "*")):
        print(seg_path)
        path_parts = os.path.normpath(seg_path).split(os.sep)
        gaussian_dir = path_parts[-3]
        plate_dir = path_parts[-2]
        file_name = path_parts[-1]
        img_save_dir = os.path.join(args.out_dir, "images", gaussian_dir, plate_dir, file_name[:-4])
        seg_save_dir = os.path.join(args.out_dir, "segmentations", gaussian_dir, plate_dir, file_name[:-4])

        os.makedirs(img_save_dir, exist_ok=True)
        os.makedirs(seg_save_dir, exist_ok=True)

        img_path = os.path.join(args.image_dir, plate_dir, file_name)

        task_args.append((img_path, seg_path, img_save_dir, seg_save_dir))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(extract_nuclei, arg) for arg in task_args]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass


if __name__ == '__main__':
    main()
