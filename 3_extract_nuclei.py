import os
from glob import glob
import tifffile
import argparse
import numpy as np
import time
import cv2
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import numpy as np
import SimpleITK as sitk
from skimage import draw
import matplotlib.pyplot as plt


def register_image_to_circle(img_np, mask_np, image_size=128, circle_radius=50):

    plt.imshow(img_np)
    plt.show()

    # Ensure image and mask are single channel and convert mask to binary format if it's not
    mask_np = (mask_np > 0).astype(np.uint8)

    # Create Circle Mask for Target Registration
    black = np.zeros((image_size, image_size))

    # Define center and radius for the disk
    center = (image_size // 2, image_size // 2)  # Center of the disk, using integer division for indices
    radius = circle_radius  # Radius of the disk

    # Generate disk coordinates
    rr, cc = draw.disk(center, radius, shape=(image_size, image_size))

    # Apply the disk to the image
    black[rr, cc] = 1  # This directly uses rr and cc for indexing

    circle_mask = black.astype('uint8')

    plt.imshow(circle_mask)
    plt.show()

    # Initialize Registration Method
    selx = sitk.ElastixImageFilter()
    selx.SetParameterMap(selx.GetDefaultParameterMap('nonrigid'))
    selx.LogToConsoleOff()

    # Convert numpy arrays back to SimpleITK Images
    movingImage = sitk.GetImageFromArray(mask_np)
    movingLabel = sitk.GetImageFromArray(img_np)
    fixedImage = sitk.GetImageFromArray(circle_mask)

    # Create Deformation Field necessary to register shape of mask to target circle
    selx.SetMovingImage(movingImage)
    selx.SetFixedImage(fixedImage)
    selx.Execute()

    # Transform Image using the deformation field from above
    resultImage = sitk.Transformix(movingLabel, selx.GetTransformParameterMap())

    # Convert result back to np.array
    registered_image_np = sitk.GetArrayFromImage(resultImage)
    registered_image_np = np.reshape(registered_image_np, (image_size, image_size))
    registered_image_np[registered_image_np < 0] = 0

    return registered_image_np.astype('uint8')


def smooth_mask(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smoothed_contours = [cv2.approxPolyDP(cnt, 0.05, True) for cnt in contours]
    mask_smoothed = np.zeros_like(mask)
    cv2.drawContours(mask_smoothed, smoothed_contours, -1, (1), thickness=cv2.FILLED)
    return mask_smoothed


def normalize_single_nuc(img):
    return (img - img.min()) / (img.max() - img.min()) * 255


def compute_orientation(mask):
    """Compute the orientation of the cell's major axis using PyTorch."""
    # Assume mask is a 2D tensor [H, W]
    y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
    y_coords, x_coords = y_coords.float(), x_coords.float()

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
    theta = 0.5 * torch.atan2(2 * mu11, mu20 - mu02)
    angle = theta * (180 / torch.pi)

    # Adjust the angle based on the orientation to ensure vertical alignment
    if angle > 0:
        angle -= 90
    elif angle < 0:
        angle += 90

    return angle.item()


def rotate_centered_cell(image, mask):
    """Rotate a centered grayscale cell image and its mask to align the major axis vertically."""
    angle = compute_orientation(mask)

    # Center of the image for rotation
    center = (image.size(1) / 2, image.size(0) / 2)

    # Rotate the image and mask
    rotated_image = TF.rotate(image.unsqueeze(0).unsqueeze(0), angle, interpolation=TF.InterpolationMode.BILINEAR,
                              center=center, expand=False)
    rotated_mask = TF.rotate(mask.unsqueeze(0).unsqueeze(0), angle, interpolation=TF.InterpolationMode.NEAREST,
                             center=center, expand=False)

    return rotated_image.squeeze(0).squeeze(0), rotated_mask.squeeze(0).squeeze(0)


def center_crop_around_com(img, mask, dim):
    # Calculate the center of mass (COM)
    y_indices, x_indices = torch.nonzero(mask, as_tuple=True)
    com_row = torch.mean(y_indices.float())
    com_col = torch.mean(x_indices.float())

    # Calculate the top-left corner of the crop based on the COM, ensuring it stays within bounds
    start_row = int(torch.clamp(com_row - dim / 2, min=0, max=mask.size(0) - dim))
    start_col = int(torch.clamp(com_col - dim / 2, min=0, max=mask.size(1) - dim))

    # Crop the mask around the COM
    cropped_mask = mask[start_row:start_row + dim, start_col:start_col + dim]
    cropped_image = img[start_row:start_row + dim, start_col:start_col + dim]

    return cropped_mask, cropped_image


def check_mass_center(mask):
    """
    Check if the mass center of the mask is at the geometric center of the image.

    Args:
    - mask (torch.Tensor): A 2D tensor representing the mask.

    Returns:
    - bool: True if the mass center is at the geometric center, False otherwise.
    - tuple: The coordinates of the mass center.
    """
    # Calculate the mass center of the mask
    y_indices, x_indices = torch.nonzero(mask, as_tuple=True)
    mass_center_x = x_indices.float().mean()
    mass_center_y = y_indices.float().mean()

    # Calculate the geometric center of the image
    image_center_x = (mask.size(1) - 1) / 2.0
    image_center_y = (mask.size(0) - 1) / 2.0

    # Determine if the mass center is close enough to the geometric center
    tolerance = 1.5  # This can be adjusted based on your requirements for "centeredness"
    is_centered = (abs(mass_center_x - image_center_x) <= tolerance) and (
            abs(mass_center_y - image_center_y) <= tolerance)

    return is_centered


def extract_cells(img_path, segmentation_path, img_out_dir, seg_out_dir, device="cpu",
                  dim=128,
                  normalize_image=True):
    if dim % 2 != 0:
        print("(!) Output image dimension must be even")
        raise ValueError()

    img_np = tifffile.imread(img_path)

    if normalize_image:
        img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255)

    mask_np = tifffile.imread(segmentation_path)

    # Define padding for later processing
    padding_val = dim // 2
    padding = (padding_val,) * 4

    # Load images and masks into PyTorch tensors
    img = torch.tensor(img_np.astype(np.uint8), device=device)
    mask = torch.tensor(mask_np.astype(np.int16), device=device)

    padded_img = F.pad(img, padding, "constant", 0)
    padded_mask = F.pad(mask, padding, "constant", 0)

    # Process each label in the mask
    unique_labels = torch.unique(mask)
    for label in unique_labels:
        if label.item() == 0:  # Skip background
            continue

        # Find the bounding box of the current nucleus
        positions = torch.where(mask == label.item())
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

        mask_to_save = roi_mask.cpu().numpy().astype(np.uint8)
        nuc_to_save = nucleus_filtered.cpu().numpy().astype(np.uint8)
        rot_nuc_to_save = nucleus_filtered_rotated.cpu().numpy().astype(np.uint8)
        rot_mask_to_save = mask_filtered_rotated.cpu().numpy().astype(np.uint8)
        norm_rot_nuc_to_save = norm_nucleus_filtered_rotated.cpu().numpy().astype(np.uint8)
        norm_nuc_to_save = norm_nucleus_filtered.cpu().numpy().astype(np.uint8)

        # Save the centered nucleus and its mask
        name = f"{label.item()}.tif"
        tifffile.imwrite(os.path.join(img_out_dir, "original_{}".format(name)),
                         nuc_to_save)
        tifffile.imwrite(os.path.join(seg_out_dir, "original_{}".format(name)),
                         mask_to_save)

        tifffile.imwrite(os.path.join(img_out_dir, "rotated_{}".format(name)),
                         rot_nuc_to_save)
        tifffile.imwrite(os.path.join(seg_out_dir, "rotated_{}".format(name)),
                         rot_mask_to_save)

        tifffile.imwrite(os.path.join(img_out_dir, "rotated_normalized_{}".format(name)),
                         norm_rot_nuc_to_save)
        tifffile.imwrite(os.path.join(img_out_dir, "original_normalized_{}".format(name)),
                         norm_nuc_to_save)


def main():
    parser = argparse.ArgumentParser(description="Cell extraction")

    # Add arguments
    parser.add_argument("--image_dir", type=str, help="Image directory path (dir that contains plate dirs)",
                        nargs='?', default="/home/christer/Datasets/MCF7/raw_extracted")
    parser.add_argument("--gpu_id", type=str, help="GPU ID", nargs='?', default="0")
    parser.add_argument("--segmentations_dir", type=str, help="Segmentations directory path",
                        nargs='?', default="/home/christer/Datasets/MCF7/structured/segmentations/gaussian_3")
    parser.add_argument("--out_dir", type=str, help="output directory",
                        nargs='?', default="/home/christer/Datasets/MCF7/structured/extracted_cells")

    # Parse arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("---->>>> DEVICE: {}".format(device))

    for seg_path in tqdm(glob(os.path.join(args.segmentations_dir, "*", "*"))):
        path_parts = os.path.normpath(seg_path).split(os.sep)
        gaussian_dir = path_parts[-3]
        plate_dir = path_parts[-2]
        file_name = path_parts[-1]
        img_save_dir = os.path.join(args.out_dir, "images", gaussian_dir, plate_dir, file_name[:-4])
        seg_save_dir = os.path.join(args.out_dir, "segmentations", gaussian_dir, plate_dir, file_name[:-4])

        os.makedirs(img_save_dir, exist_ok=True)
        os.makedirs(seg_save_dir, exist_ok=True)

        img_path = os.path.join(args.image_dir, plate_dir, file_name)

        extract_cells(img_path, seg_path, img_save_dir, seg_save_dir, device)


if __name__ == '__main__':
    main()
