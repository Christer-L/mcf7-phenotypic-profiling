import os
import pandas as pd
import argparse
from glob import glob
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap
import seaborn as sns

from catboost import Pool, cv
from sklearn.model_selection import StratifiedShuffleSplit
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix


def format_metadata(metadata_path, batch_dir):
    paths_list = []
    for filename in glob(batch_dir + '/*'):
        with open(filename, 'rb') as f:
            dapi_paths = pickle.load(f)
            paths_list.append(dapi_paths)
    paths = [p.split("/")[1] + ".tif" for p in np.concatenate(paths_list)]
    metadata = pd.read_csv(metadata_path)
    filtered_metadata = metadata[metadata["Image_FileName_DAPI"].isin(paths)]
    return filtered_metadata


#def normalize_and_plot():

def main():
    parser = argparse.ArgumentParser(description="Profile formatting")

    # Add arguments
    parser.add_argument("--mevae_dir", type=str, help="Directory containing outputs",
                        nargs='?',
                        default="/home/christer/Datasets/MCF7/structured/MEVAE_data/Output_0-001kl_loss")
    parser.add_argument("--profile_dir", type=str, help="Directory containing profiles",
                        nargs='?',
                        default="/home/christer/Datasets/MCF7/structured/profiles/")
    parser.add_argument("--batch_dir", type=str, help="Directory containing dapi paths (pickle files)",
                        nargs='?',
                        default="/home/christer/Datasets/MCF7/structured/metadata/batches")
    parser.add_argument("--metadata", type=str, help="metadata table",
                        nargs='?',
                        default="/home/christer/Datasets/MCF7/structured/metadata/metadata.csv")
    parser.add_argument("--out_dir", type=str, help="Output directory for profiles",
                        nargs='?', default="/home/christer/Datasets/MCF7/out")
    parser.add_argument("--gaussian", type=str, help="Gaussian kernel sigma used in segmentation",
                        nargs='?', default="3")
    parser.add_argument("--moas", type=str, help="Moa file",
                        nargs='?', default="/home/christer/Datasets/MCF7/structured/metadata/moas.csv")

    args = parser.parse_args()

    metadata = format_metadata(args.metadata, args.batch_dir)
    moas = pd.read_csv(args.moas)
    moas_ = []
    drugs = []

    mevae_paths = pd.read_csv(os.path.join(args.mevae_dir, "filenames.csv"), header=None)[0].tolist()
    mevae_paths = np.array(["_".join(os.path.basename(path).split("_")[2:-1]) for path in mevae_paths])

    encodings = np.array(pd.read_csv(os.path.join(args.mevae_dir, "encodings.csv"), header=None))

    image_profiles = []
    pyrad_profiles = []
    dino_profiles = []
    mevae_profiles = []
    for i, row in tqdm(metadata.iterrows()):
        name = row["Image_FileName_DAPI"][:-4]
        directory = row["Image_PathName_DAPI"].split("/")[1]

        idx_mevae = np.where(mevae_paths == name)
        encs_mevae = pd.DataFrame(encodings[idx_mevae], columns=["{}_MEVAE".format(i) for i in range(64)])
        mevae_profile = encs_mevae.mean().to_frame().T
        mevae_profiles.append(mevae_profile)

        dino_profile_path = os.path.join(args.profile_dir,
                                         "dino",
                                         "gaussian_{}".format(args.gaussian),
                                         directory,
                                         name + ".csv")
        pyrad_profile_path = os.path.join(args.profile_dir,
                                          "pyradiomics",
                                          "gaussian_{}".format(args.gaussian),
                                          directory,
                                          name + ".csv")
        drug = row["Image_Metadata_Compound"]
        moa = moas[moas["compound"] == drug]["moa"].tolist()[0]

        pyrad_profile = pd.read_csv(pyrad_profile_path)
        pyrad_profile_filtered = pyrad_profile[pyrad_profile["normalized"] == True].drop(
            columns=["Unnamed: 0", "id", "normalized"]).mean().to_frame().T
        dino_profile = pd.read_csv(dino_profile_path)
        dino_profile_filtered = dino_profile[dino_profile["normalized"] == True].drop(
            columns=["Unnamed: 0", "id", "normalized"]).mean().to_frame().T
        drugs.append(drug)
        moas_.append(moa)
        joint_profile = pd.concat([pyrad_profile_filtered, dino_profile_filtered], axis=1)
        image_profiles.append(joint_profile)
        pyrad_profiles.append(pyrad_profile_filtered)
        dino_profiles.append(dino_profile_filtered)

    mevae_profiles = pd.concat(mevae_profiles)
    py_di_profiles = pd.concat(image_profiles)
    pyrad_profiles = pd.concat(pyrad_profiles)
    dino_profiles = pd.concat(dino_profiles)
    dino_profiles = (dino_profiles - dino_profiles.mean()) / dino_profiles.std()
    pyrad_profiles = (pyrad_profiles - pyrad_profiles.mean()) / pyrad_profiles.std()

    py_di_profiles = (py_di_profiles - py_di_profiles.mean()) / py_di_profiles.std()
    image_profiles = pd.concat([py_di_profiles, mevae_profiles], axis=1)

    image_profiles["drug"] = drugs
    image_profiles["moa"] = moas_

    profiles = image_profiles.reset_index().drop(columns=["index"])
    profiles.to_csv(os.path.join(args.out_dir, "profiles_all.csv"))

    reducer = umap.UMAP(min_dist=0.3, n_neighbors=45)

    y_values = profiles["drug"]
    y_values2 = profiles["moa"]
    x_values = profiles.drop(columns=["drug", "moa", "original_firstorder_Maximum"]).values

    #scaled_data = StandardScaler().fit_transform(x_values)
    embedding = reducer.fit_transform(x_values)

    sns.set_theme(rc={'figure.figsize': (20, 10)})
    ax = sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        style=y_values,
        hue=y_values2, s=40)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(os.path.join(args.out_dir, "all.svg"), bbox_inches='tight')
    plt.close()

    sns.countplot(x=y_values2)
    plt.savefig(os.path.join(args.out_dir, "countplot.svg"), bbox_inches='tight')

    c = np.unique(np.array(y_values2), return_counts=True)[1].min()

    profiles = profiles.groupby('moa').apply(lambda s: s.sample(c, replace=True))
    x_values_filtered = profiles.drop(columns=["drug", "moa", "original_firstorder_Maximum"])
    y_values_filtered = profiles["moa"].values

    test = x_values_filtered.columns

    profiles_dino = x_values_filtered.drop(
        columns=[col for col in x_values_filtered.columns.tolist() if ("MEVAE" in col) or ("original" in col)])
    profiles_pyrad = x_values_filtered.drop(
        columns=[col for col in x_values_filtered.columns.tolist() if "original" not in col])
    profiles_mevae = x_values_filtered.drop(
        columns=[col for col in x_values_filtered.columns.tolist() if col[3:] != "MEVAE"])

    print("----------- ALL")
    cv_dataset = Pool(data=x_values_filtered.values,
                      label=y_values_filtered)

    params = {"iterations": 200,
              "depth": 4,
              "loss_function": "MultiClass",
              "eval_metric": "Accuracy",
              "verbose": False}

    scores = cv(cv_dataset,
                params,
                fold_count=3,
                plot="True")

    print("---------- DINO")
    cv_dataset = Pool(data=profiles_dino.values,
                      label=y_values_filtered)

    params = {"iterations": 200,
              "depth": 4,
              "loss_function": "MultiClass",
              "eval_metric": "Accuracy",
              "verbose": False}

    scores = cv(cv_dataset,
                params,
                fold_count=3,
                plot="True")

    print("---------- MEVAE")
    cv_dataset = Pool(data=profiles_mevae.values,
                      label=y_values_filtered)

    params = {"iterations": 200,
              "depth": 4,
              "loss_function": "MultiClass",
              "eval_metric": "Accuracy",
              "verbose": False}

    scores = cv(cv_dataset,
                params,
                fold_count=3,
                plot="True")

    print("---------- PYRAD")
    cv_dataset = Pool(data=profiles_pyrad.values,
                      label=y_values_filtered)

    params = {"iterations": 200,
              "depth": 4,
              "loss_function": "MultiClass",
              "eval_metric": "Accuracy",
              "verbose": False}

    scores = cv(cv_dataset,
                params,
                fold_count=3,
                plot="True")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

    for train_index, test_index in sss.split(x_values_filtered, y_values_filtered):
        X_train, X_test = x_values_filtered[train_index], x_values_filtered[test_index]
        y_train, y_test = y_values_filtered[train_index], y_values_filtered[test_index]

    # Initialize and train a CatBoostClassifier
    model = CatBoostClassifier(iterations=100, depth=3, loss_function='MultiClass', verbose=False)
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    predictions = model.predict(X_test)

    # Compute the confusion matrix for this fold and add it to the list
    cm = confusion_matrix(y_test, predictions, labels=model.classes_)

    # Class labels for the confusion matrix
    class_labels = model.classes_  # Adjust as per your classes

    plt.figure(figsize=(10, 7))  # Size of the figure
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels, rotation=45, va="center")

    # Adding labels and title
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.title('Confusion Matrix')

    plt.savefig(os.path.join(args.out_dir, "confusion_matrix.svg"), bbox_inches='tight')

    #
    #
    # # -----
    #
    # profiles = profiles.reset_index().drop(columns=["index"])
    #
    # reducer = umap.UMAP(min_dist=0.3, n_neighbors=45)
    #
    # y_values = profiles["drug"]
    # y_values2 = profiles["moa"]
    # x_values = profiles.drop(columns=["drug", "moa"]).values
    #
    # scaled_data = StandardScaler().fit_transform(x_values)
    # embedding = reducer.fit_transform(scaled_data)
    #
    # sns.set_palette(sns.color_palette("Spectral"))
    #
    # sns.scatterplot(
    #     x=embedding[:, 0],
    #     y=embedding[:, 1],
    #     hue=y_values, s=3)
    # plt.show()
    #
    # sns.set_palette(sns.color_palette("Spectral"))
    #
    # sns.scatterplot(
    #     x=embedding[:, 0],
    #     y=embedding[:, 1],
    #     hue=y_values2, s=3)
    # plt.show()


if __name__ == '__main__':
    main()
