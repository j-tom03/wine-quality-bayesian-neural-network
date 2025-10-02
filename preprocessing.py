import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def remove_duplicate_data(df):
    df.drop_duplicates(inplace=True)

    return df

def scale_features(df):
    X, y = df.drop("quality", axis="columns"), df["quality"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=df.index)

    X_scaled["quality"] = y.values

    return X_scaled, scaler

def balance_classes(df, colour=None, smote_sampling="auto"):
    X, y = df.drop("quality", axis="columns"), df["quality"]

    smote = SMOTE(random_state=19, k_neighbors=4, sampling_strategy=smote_sampling)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled["quality"] = y_resampled

    if colour is not None:
        sns.barplot(x=df_resampled["quality"].value_counts().index,
                y=df_resampled["quality"].value_counts().values,
                palette="viridis")
        
        plt.xlabel("Quality")
        plt.ylabel("Count")
        plt.title(f"Resampled Quality Class Distribution - {colour} Wine")
        plt.savefig(f"./plots/resampled_{colour}_quality_dist_plot.png")
        plt.show()

    return df_resampled

def remove_outliers(df, threshold_z=4):
    df = df.reset_index(drop=True)
    z_scores = np.abs((df - df.mean()) / df.std())
    outlier_indices = np.where(z_scores > threshold_z)[0]
    df_clean = df.drop(index=outlier_indices)

    return df_clean

def preprocess(df, colour):
    df = remove_duplicate_data(df)
    df, scaler = scale_features(df)
    df = remove_outliers(df)
    df = balance_classes(df)

    return df