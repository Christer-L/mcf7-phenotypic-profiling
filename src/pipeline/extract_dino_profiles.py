import argparse
import pickle

import pandas as pd
import tifffile
from tqdm import tqdm
import torch
import os

from glob import glob
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

# Amount of dimensions in the DINOv2 small model
EMBEDDING_DIM = 384


def get_dino_embedding(processor, model, img, gpu_id):
    """
    Compute DINOv2 embeddings for a given image object.
    Model and processor can be initialized with the initialize_dino function.

    Parameters:
    processor (AutoImageProcessor): The image processor from the DINOv2 model.
    model (AutoModel): The DINOv2 model.
    img (np.ndarray): The image to compute embeddings for.
    gpu_id (str): The ID of the GPU to use for computation.

    Returns:
    np.ndarray: The predicted DINOv2 embeddings for the image.
    """
    # Convert the image to a PIL Image object and process it using the DINO model's processor
    inputs = processor(images=Image.fromarray(img), return_tensors="pt")

    # Move the processed image tensor to the GPU specified by `gpu_id`
    inputs = {name: tensor.to('cuda:{}'.format(gpu_id)) for name, tensor in inputs.items()}

    # Pass the processed image tensor through the DINO model to get the model's outputs
    outputs = model(**inputs)

    # Extract the last hidden state from the model's outputs
    last_hidden_state = outputs.last_hidden_state

    # Compute the mean over patches to get the DINOv2 embeddings
    embeddings = last_hidden_state.mean(dim=1)

    return embeddings.cpu().detach().numpy()


def initialize_dino(gpu_id):
    """
    Initialize the DINOv2 model and its processor and load the model to the specified GPU.

    Parameters:
    gpu_id (str): The ID of the GPU to use for computation.

    Returns:
    tuple: A tuple containing the processor and the model.
    """
    # Load the DINOv2 model and its processor from the pretrained 'facebook/dinov2-small' model
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
    model = AutoModel.from_pretrained('facebook/dinov2-small')

    # Move the DINOv2 model to the GPU specified by `gpu_id`
    model = model.to('cuda:{}'.format(gpu_id))

    return processor, model


def main():
    """
    Main function to extract DINOv2 profiles from images of single objects.
    """

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="DINO profile extraction")

    # Define command line arguments
    parser.add_argument("--batch_dir", type=str, help="Path to the folder with pickle files",
                        nargs='?',
                        default="/mnt/cbib/christers_data/mcf7/structured/metadata/drug_selection/paths")
    parser.add_argument("--batch", type=str, help="Batch number",
                        nargs='?',
                        default="0")
    parser.add_argument("--image_dir", type=str, help="extracted_cells directory",
                        nargs='?',
                        default="/mnt/cbib/christers_data/mcf7/structured/extracted_cells")
    parser.add_argument("--gpu_id", type=str, help="GPU ID", nargs='?', default="7")
    parser.add_argument("--out_dir", type=str, help="output directory for dino profiles",
                        nargs='?', default="/mnt/cbib/christers_data/mcf7/structured/profiles/dino")
    parser.add_argument("--gaussian", type=str, help="Gaussian kernel sigma used in segmentation",
                        nargs='?', default="3")

    # Parse command line arguments
    args = parser.parse_args()

    # Load paths from a pickle file
    pickle_path = os.path.join(args.batch_dir, "batch_{}.pkl".format(args.batch))
    with open(pickle_path, 'rb') as f:
        paths = pickle.load(f)

    # Initialize the DINOv2 model
    processor, model = initialize_dino(args.gpu_id)

    # Iterate over the paths
    for dir_path in tqdm(paths):
        path_parts = os.path.normpath(dir_path).split(os.sep)
        columns = ["object_id"] + ["DINOv2_{}".format(i) for i in range(EMBEDDING_DIM)]
        rows = []

        # Construct the path to the image stack
        img_stack_path = os.path.join(args.image_dir,
                                      "images",
                                      "gaussian_{}".format(args.gaussian),
                                      dir_path,
                                      "rotated_normalized.tif")

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
        save_path = os.path.join(args.out_dir, "gaussian_{}".format(args.gaussian), path_parts[0])
        os.makedirs(save_path, exist_ok=True)
        save_filepath = os.path.join(save_path, "{}.csv".format(path_parts[1]))

        # Save the DataFrame to a CSV file
        profile_df.to_csv(save_filepath)


if __name__ == '__main__':
    main()
