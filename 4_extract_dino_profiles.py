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

EMBEDDING_DIM = 384


def get_dino_embedding(processor, model, cell_img):
    inputs = processor(images=Image.fromarray(cell_img), return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    embeddings = last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()


def initialize_dino():
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
    model = AutoModel.from_pretrained('facebook/dinov2-small')
    return processor, model


def main():
    parser = argparse.ArgumentParser(description="DINO profile extraction")

    # Add arguments
    parser.add_argument("--pathnames", type=str, help="Pickle file with metadata",
                        nargs='?',
                        default="/home/christer/Datasets/MCF7/structured/metadata/batches"
                                "/filtered_drugs_dapi_dirs_batch_0.pkl")
    parser.add_argument("--image_dir", type=str, help="extracted_cells image dir",
                        nargs='?',
                        default="/home/christer/Datasets/MCF7/structured/extracted_cells/images"
                        )
    parser.add_argument("--gpu_id", type=str, help="GPU ID", nargs='?', default="7")
    parser.add_argument("--out_dir", type=str, help="output directory for profiles",
                        nargs='?', default="/home/christer/Datasets/MCF7/structured/profiles/dino")
    parser.add_argument("--gaussian", type=str, help="Gaussian kernel sigma used in segmentation",
                        nargs='?', default="3")

    # Parse arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("---->>>> DEVICE: {}".format(device))

    with open(args.pathnames, 'rb') as f:
        paths = pickle.load(f)

    processor, model = initialize_dino()

    for dir_path in tqdm(paths):
        path_parts = os.path.normpath(dir_path).split(os.sep)
        columns = ["id", "normalized"] + list(range(EMBEDDING_DIM))
        rows = []
        for condition in ["rotated_normalized_*.tif", "rotated_*.tif"]:
            for img_path in glob(os.path.join(args.image_dir, "gaussian_{}".format(args.gaussian),
                                              dir_path,
                                              "rotated_normalized_*.tif")):
                normalized = condition == "rotated_normalized_*.tif"

                idx = img_path[:-4].split("_")[-1]

                img = tifffile.imread(img_path)
                embedding = get_dino_embedding(processor, model, img)
                entry = [int(idx), normalized] + embedding[0].tolist()
                rows.append(entry)

        profile_df = pd.DataFrame(rows, columns=columns)
        save_path = os.path.join(args.out_dir, "gaussian_{}".format(args.gaussian), path_parts[0])
        os.makedirs(save_path, exist_ok=True)
        save_filepath = os.path.join(save_path, "{}.csv".format(path_parts[1]))
        profile_df.to_csv(save_filepath)


if __name__ == '__main__':
    main()
