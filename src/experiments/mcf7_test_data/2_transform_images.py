import argparse
import os
from glob import glob
from skimage.data import data_dir
import tifffile
from tqdm import tqdm
import concurrent
from concurrent.futures import ProcessPoolExecutor
import traceback

from src.pipeline.normalize_object_shape import transform_image

def process_image(args):
    img_path, mask_path, out_path = args
    print("started converting ", img_path, " with out dir: ", out_path)
    traceback.print_exc()

    if os.path.isfile(img_path) and os.path.isfile(mask_path):
        print(img_path, " exists")
        traceback.print_exc()

        img = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)

        transformed_img = transform_image(img, mask, image_size=179, circle_radius=89)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tifffile.imwrite(out_path, transformed_img)

        print("saved ", img_path, " to ", out_path)
        traceback.print_exc()
    else:
        print("no ", img_path)
        traceback.print_exc()

def main():
    # Arguments
    parser = argparse.ArgumentParser(description="Format image paths")

    parser.add_argument("data_dir", type=str,
                        help="Directories with segmentation maps",
                        nargs="?",
                        default="/mnt/cbib/pnria/profiling_test_data/structured")

    parser.add_argument("out_dir", type=str,
                        help="Output directory",
                        nargs="?",
                        default="/mnt/cbib/christers_data/test_mcf7")

    # Parse arguments
    args = parser.parse_args()
    data_dir = args.data_dir
    out_dir = args.out_dir

    stack_paths = glob(os.path.join(data_dir, "images", "*", "*"))

    task_args = []

    for stack_path in stack_paths:
        path_parts = stack_path.split(os.sep)
        img_id = path_parts[-1]
        pattern_template = path_parts[-2]

        original_img_path = os.path.join(data_dir, "images", pattern_template, img_id, "original.tif")
        rotated_img_path = os.path.join(data_dir, "images", pattern_template, img_id, "rotated.tif")

        original_seg_path = os.path.join(data_dir, "segmentations", pattern_template, img_id, "original.tif")
        rotated_seg_path = os.path.join(data_dir, "segmentations", pattern_template, img_id, "rotated.tif")

        save_path_original = os.path.join(out_dir, pattern_template, img_id, "original.tif")
        save_path_rotated = os.path.join(out_dir, pattern_template, img_id, "rotated.tif")

        task_args.append((original_img_path, original_seg_path, save_path_original))
        task_args.append((rotated_img_path, rotated_seg_path, save_path_rotated))

        print("added ", original_img_path)
        traceback.print_exc()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, arg) for arg in task_args]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

if __name__ == '__main__':
    main()