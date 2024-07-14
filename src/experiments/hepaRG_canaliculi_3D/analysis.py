import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def remove_um_units(df):
    concentrations = df["concentration"]
    correct_concentrations = [float(conc[:-2]) for conc in concentrations]
    df["concentration"] = correct_concentrations
    return df


def main():
    df = pd.read_csv('/mnt/cbib/christers_data/organoids/canaliculi_analysis/all_profiles.csv')
    cpz_df = df[(df["condition"] == "CPZ") | (df["condition"] == "Control")].drop(["condition", "path"], axis=1)
    ind_df = df[(df["condition"] == "IND") | (df["condition"] == "Control")].drop(["condition", "path"], axis=1)
    ind_df = remove_um_units(ind_df)
    cpz_df = remove_um_units(cpz_df)

    for col in ind_df.drop(["concentration", "object_id"], axis=1).columns:
        g = sns.catplot(
            data=ind_df, x="concentration", y=col,
            capsize=.2, errorbar="se",
            kind="point", height=6, aspect=.75,)
        g.despine(left=True)
        plt.show()
        g.savefig(os.path.join("/mnt/cbib/christers_data/organoids/out_canaliculi_plots/IND", f"ind_{col}.png"))

    for col in cpz_df.drop(["concentration", "object_id"], axis=1).columns:
        g = sns.catplot(
            data=cpz_df, x="concentration", y=col,
            capsize=.2, errorbar="se",
            kind="point", height=6, aspect=.75, )
        g.despine(left=True)
        plt.show()
        g.savefig(os.path.join("/mnt/cbib/christers_data/organoids/out_canaliculi_plots/CPZ", f"ind_{col}.png"))


    print()


if __name__ == '__main__':
    main()
