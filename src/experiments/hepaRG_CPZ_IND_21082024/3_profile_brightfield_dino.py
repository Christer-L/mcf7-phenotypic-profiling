from src.pipeline.extract_dino_profiles import get_dino_embedding, initialize_dino
import argparse
import os
from tqdm import tqdm
import tifffile
import pandas as pd
from glob import glob


# Amount of dimensions in the DINOv2 small model
EMBEDDING_DIM = 384


def main():
    """
    Main function to extract DINOv2 profiles from images of single objects.
    """

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="DINO profile extraction")

    # Define command line arguments
    parser.add_argument("--data_dir", type=str, help="extracted_objects directory",
                        nargs='?',
                        default="/mnt/cbib/christers_data/HepaRG/extracted_brightfield_objects/HepaRG_CPZ_IND_21082024")

    parser.add_argument("--out_dir", type=str, help="output directory for dino profiles",
                        nargs='?', default="/mnt/cbib/christers_data/HepaRG/brightfield_profiles/"
                                           "dino/HepaRG_CPZ_IND_21082024")

    parser.add_argument("--gpu_id", type=str, help="GPU ID", nargs='?', default="6")

    # Parse command line arguments
    args = parser.parse_args()
    data_dir = args.data_dir
    out_dir = args.out_dir
    gpu_id = args.gpu_id

    paths = glob(os.path.join(data_dir, "images", "*", "Fixed", "*", "*_w1Brightfield"))

    # Initialize the DINOv2 model
    processor, model = initialize_dino(gpu_id)

    # Iterate over the paths
    for dir_path in tqdm(paths):
        path_parts = dir_path.split(os.sep)[8:]
        columns = ["object_id"] + ["DINOv2_{}".format(i) for i in range(EMBEDDING_DIM)]
        rows = []

        # Construct the path to the image stack
        img_stack_path = os.path.join(dir_path, "rotated_normalized.tif")

        # Read the image stack
        img_stack = tifffile.imread(str(img_stack_path))

        n_objects = img_stack.shape[0]

        # For each image in the stack, compute the DINOv2 embedding
        for idx in range(n_objects):
            img = img_stack[idx]
            embedding = get_dino_embedding(processor, model, img, args.gpu_id)
            entry = [int(idx)] + embedding[0].tolist()
            rows.append(entry)

        # Append the embeddings to a DataFrame
        profile_df = pd.DataFrame(rows, columns=columns)

        # Construct the path to the output CSV file
        save_path = os.path.join(args.out_dir, *path_parts[:-1])
        os.makedirs(save_path, exist_ok=True)
        save_filepath = os.path.join(save_path, "{}.csv".format(str(path_parts[-1])))
        print(save_filepath)

        # Save the DataFrame to a CSV file
        profile_df.to_csv(save_filepath)


if __name__ == '__main__':
    main()
