import argparse
from glob import glob

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os
import umap
import pandas as pd
import hdbscan
from bokeh.plotting import figure, show, save
from bokeh.io import output_notebook, output_file
from bokeh.models import ColumnDataSource, HoverTool
from sklearn.cluster import HDBSCAN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from catboost import CatBoostClassifier, Pool
import tifffile
import matplotlib.image as mpimg
import cv2

import matplotlib

cmap = matplotlib.cm.get_cmap('autumn')


def remove_boundary_touching_slices(seg_stack):
    # Initialize a list to store the filtered slices
    filtered_slices = []

    for i, seg in enumerate(seg_stack):
        # Find contours in the segmentation mask
        contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Flag to determine if any contour touches the boundary
        touches_boundary = False

        for cnt in contours:
            if any([pt[0][0] == 0 or pt[0][0] == seg.shape[1] - 1 or pt[0][1] == 0 or pt[0][1] == seg.shape[0] - 1 for
                    pt in cnt]):
                touches_boundary = True
                break

        # If no object touches the boundary, add the slice to the filtered list
        if not touches_boundary:
            filtered_slices.append(seg)

    # Convert the filtered list back to a numpy array
    return np.array(filtered_slices)


def highlight_segmentations(img, seg_stack, spheroid_idx, color):
    # Convert the image to BGR if it is grayscale
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Ensure the image is in the correct dtype
    if img.dtype != np.uint8:
        # minmax norm image
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

    img_raw = img.copy()

    # Iterate through each slice in the seg_stack
    for i, seg in enumerate(seg_stack):
        # Convert segmentation mask to the correct type if necessary
        if seg.dtype != np.uint8:
            seg = seg.astype(np.uint8)

        # Ensure segmentation mask is binary
        seg = cv2.threshold(seg, 127, 255, cv2.THRESH_BINARY)[1]

        # Find contours in the segmentation mask
        contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Choose the color based on the slice index
        if i == spheroid_idx:
            #color = (255, 0, 0) if i == spheroid_idx else (0, 0, 255)
            mask = np.zeros_like(img)
            cv2.fillPoly(mask, contours, color[:3])

            img = cv2.addWeighted(img, 0.7, mask, 0.3, 0)

    plt.imshow(img)
    plt.show()

    return img, img_raw


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="DINO profile extraction")

    # Define command line arguments
    parser.add_argument("--profiles_path", type=str, help="extracted_objects directory",
                        nargs='?',
                        default="/Users/christer/Temp/final_profiles.csv")

    # Parse command line arguments
    args = parser.parse_args()
    profiles = pd.read_csv(args.profiles_path)
    profiles = profiles.drop(columns=['Pyradiomics_original_firstorder_Maximum'])

    norm_profiles = profiles.apply(
        lambda x: (x - x.mean()) / x.std() if x.name not in ['object_id', 'image_id', 'condition'] else x)

    # Keep only cols with pyradiomics or DINOv2 in their name
    features = norm_profiles.filter(regex='Pyradiomics|DINOv2').values

    img_paths = ["/Users/christer/Temp/extracted_spheroids/{}_image{}_object{}.png".format(r["condition"],
                                                                                           r["image_id"],
                                                                                           r["object_id"])
                 for _, r in profiles.iterrows()]


    reducer = umap.UMAP()
    embedding = reducer.fit_transform(norm_profiles.filter(regex='Pyradiomics|DINOv2').values)
    df_embedding = pd.DataFrame(embedding, columns=['x', 'y'])

    hdb = HDBSCAN(min_cluster_size=20)
    hdb.fit(df_embedding.values)
    print(np.unique(hdb.labels_))
    profiles["cluster"] = hdb.labels_
    profiles["img_path"] = img_paths

    source = ColumnDataSource(
        data=dict(
            x=df_embedding["x"].values,
            y=df_embedding["y"].values,
            imgs=profiles["img_path"].values,
            color=[x for x in profiles.cluster.map({3: "red", 0: "blue", 1: "green", 2: "black"})],
        )
    )

    hover = HoverTool(
        tooltips="""
                    <div>
                        <div>
                            <img
                                src="@imgs" height="256" alt="Image not available" width="256"
                                style="float: left; margin: 0px 15px 15px 0px;"
                                border="2"
                            ></img>
                        </div>
                    <...other div tags for text>
                    """
    )

    # Creating a figure
    p = figure(tools=[hover, 'pan', 'box_zoom'])
    # Draw the plot using column data source
    p.scatter('x', 'y', color='color', source=source)
    output_file("plot_umap_clusters.html", title="UMAP Brightfield clusters (HDBSCAN)")
    save(p)

    print()
    for i_clust in range(4):
        cluster_profiles = profiles[profiles.cluster == i_clust]
        print(len(cluster_profiles))
        if len(cluster_profiles) == 137:
            break

    norm_cluster_profiles = cluster_profiles.apply(
        lambda x: (x - x.mean()) / x.std() if x.name not in ['object_id', 'image_id', 'condition', 'cluster', 'img_path'] else x)

    mean_profile = norm_cluster_profiles.filter(regex='Pyradiomics|DINOv2').mean()

    vals_drug, counts_drug = np.unique(norm_cluster_profiles[
                                           norm_cluster_profiles.condition == "filtered_100"].image_id.values,
                                       return_counts=True)

    euc_dist = lambda x, y: sum([(x.iloc[i] - y.iloc[i]) ** 2 for i in range(len(x))]) ** 0.5

    for i_val in range(len(vals_drug)):
        val = vals_drug[i_val]
        count = counts_drug[i_val]
        if count > 1:
            profiles_val = norm_cluster_profiles[(norm_cluster_profiles.image_id == val) &
                                            (norm_cluster_profiles.condition == "filtered_100")]
            profile_dist = []

            for idx, r in profiles_val.filter(regex="Pyradiomics|DINOv2").iterrows():
                profile_dist.append(euc_dist(r, mean_profile))
            profiles_val["dist"] = profile_dist
            keep_row_index = profiles_val.loc[profiles_val['dist'].idxmin()].name
            profiles_val.drop(keep_row_index, inplace=True)
            to_drop = profiles_val.index
            cluster_profiles.drop(to_drop, inplace=True)

    vals_drug, counts_drug = np.unique(norm_cluster_profiles[
                                           norm_cluster_profiles.condition == "filtered_ctrl"].image_id.values,
                                       return_counts=True)

    euc_dist = lambda x, y: sum([(x.iloc[i] - y.iloc[i]) ** 2 for i in range(len(x))]) ** 0.5

    for i_val in range(len(vals_drug)):
        val = vals_drug[i_val]
        count = counts_drug[i_val]
        if count > 1:
            profiles_val = norm_cluster_profiles[(norm_cluster_profiles.image_id == val) &
                                                 (norm_cluster_profiles.condition == "filtered_ctrl")]
            profile_dist = []

            for idx, r in profiles_val.filter(regex="Pyradiomics|DINOv2").iterrows():
                profile_dist.append(euc_dist(r, mean_profile))
            profiles_val["dist"] = profile_dist
            keep_row_index = profiles_val.loc[profiles_val['dist'].idxmin()].name
            profiles_val.drop(keep_row_index, inplace=True)
            to_drop = profiles_val.index
            cluster_profiles.drop(to_drop, inplace=True)

    # for c in cluster_profiles.filter(regex='Pyradiomics_original_shape').columns:
    #     print(c)
    #     sns.kdeplot(x=c,
    #                 data=cluster_profiles[cluster_profiles["condition"] == "filtered_ctrl"], color="green")
    #     sns.kdeplot(x=c,
    #                 data=cluster_profiles[cluster_profiles["condition"] == "filtered_100"], color="purple")
    #
    #     plt.show()

    # for idx, row in cluster_profiles[cluster_profiles["condition"] == "filtered_100"].iterrows():
    #     img_p = "/Users/christer/Temp/all_spheroids/drug/{}.png".format(idx)
    #     img = mpimg.imread("/Users/christer/Temp/extracted_spheroids/{}_image{}_object{}.png".format(
    #         row["condition"],
    #         row["image_id"],
    #         row["object_id"]))
    #     mpimg.imsave(img_p, img)
    #
    # for idx, row in cluster_profiles[cluster_profiles["condition"] == "filtered_ctrl"].iterrows():
    #     img_p = "/Users/christer/Temp/all_spheroids/ctrl/{}.png".format(idx)
    #     img = mpimg.imread("/Users/christer/Temp/extracted_spheroids/{}_image{}_object{}.png".format(
    #         row["condition"],
    #         row["image_id"],
    #         row["object_id"]))
    #     mpimg.imsave(img_p, img)
    print()

        # cluster_x = cluster_profiles.filter(regex='Pyradiomics_original_shape').values
        # X_train, X_test, y_train, y_test = train_test_split(cluster_x, cluster_profiles["condition"].to_list(), test_size = 0.2, random_state = 42, stratify=cluster_profiles["cluster"])
        #
        # model = CatBoostClassifier(iterations=100,
        #                            learning_rate=1,
        #                            depth=2,
        #                            eval_metric='Accuracy',)
        #
        # train_data = Pool(data=X_train, label=y_train)
        # eval_data = Pool(data=X_test, label=y_test)
        #
        # model.fit(train_data, eval_set=eval_data)


        # clf = LinearDiscriminantAnalysis()
        # clf.fit(X_train, y_train)
        # df_embedding = pd.DataFrame(embedding, columns=["x", "y"])
    #
    #     source = ColumnDataSource(
    #         data=dict(
    #             x=df_embedding["x"].values,
    #             y=df_embedding["y"].values,
    #             imgs=cluster_profiles["img_path"].values,
    #             color=[x for x in cluster_profiles.condition.map({"filtered_ctrl": "red", "filtered_100": "blue"})]
    #         )
    #     )
    #
    #     hover = HoverTool(
    #         tooltips="""
    #                     <div>
    #                         <div>
    #                             <img
    #                                 src="@imgs" height="256" alt="Image not available" width="256"
    #                                 style="float: left; margin: 0px 15px 15px 0px;"
    #                                 border="2"
    #                             ></img>
    #                         </div>
    #                     <...other div tags for text>
    #                     """
    #     )
    #
    #     # Creating a figure
    #     p = figure(tools=[hover, 'pan', 'box_zoom'])
    #     # Draw the plot using column data source
    #     p.scatter('x', 'y', color='color', source=source)
    #     output_file("plot_umap_experimental_{}.html".format(int(l+2)), title="Some title")
    #     save(p)
    #
    #
    #
    #
    # # plt.scatter(
    # #     embedding[:, 0],
    # #     embedding[:, 1],
    # #     s=2,
    # #     c=[sns.color_palette()[x] for x in profiles.condition.map({"filtered_ctrl": 0, "filtered_100": 1})])

    features = cluster_profiles.filter(regex="Pyradiomics|DINOv2")
    features_norm = features.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    cluster_profiles[features.columns] = features_norm
    cluster_profiles["rank"] = cluster_profiles["Pyradiomics_original_shape2D_PixelSurface"].rank(ascending=False)
    empty_images = 1000

    for condition_i, condition_dir in enumerate(glob("/Users/christer/Temp/segmentations/*")):
        condition = os.path.basename(condition_dir)
        img_path = "/Users/christer/Temp/sanofi_presentation/{}.tif".format(condition)
        img = tifffile.imread(img_path)
        for rank, seg_path in enumerate(glob(os.path.join(condition_dir, "*"))):
            img_idx = str(os.path.basename(seg_path)[:-4])
            img_slice = img[int(img_idx)]
            seg_stack = remove_boundary_touching_slices(tifffile.imread(seg_path))
            spheroid_idx_rows = cluster_profiles[(cluster_profiles["image_id"] == int(img_idx))
                                            & (cluster_profiles["condition"] == condition)]

            spheroid_idx = spheroid_idx_rows["object_id"].values
            if len(spheroid_idx) == 0:
                area_norm = 0
                #rank = empty_images + 1
                empty_images = empty_images + 1
            else:
                area_norm = spheroid_idx_rows["Pyradiomics_original_shape2D_PixelSurface"].values[0]
                #rank = spheroid_idx_rows["rank"].values[0]
            print(len(spheroid_idx))

            color_spheroid = [round(i * 255) for i in cmap(area_norm)]
            print(color_spheroid)
            if len(spheroid_idx) != 0:
                spheroid_idx = spheroid_idx[0]
                highlighted_img, img_raw = highlight_segmentations(img_slice, seg_stack, int(spheroid_idx), color_spheroid)
            else:
                highlighted_img, img_raw = highlight_segmentations(img_slice, seg_stack, 1000000, color_spheroid)
            tifffile.imwrite("/Users/christer/Temp/out3/autumn_cmap/{}/{}_{}.tiff".format(condition, int(rank), img_idx), highlighted_img)
            #tifffile.imwrite("/Users/christer/Temp/out2/images/{}/{}_{}.tiff".format(condition, int(rank), img_idx), img_raw)


if __name__ == '__main__':
    main()
