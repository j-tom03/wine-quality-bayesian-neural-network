import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def plot_cov(df, colour):
    df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
    cov_matrix = df.cov()

    plt.figure(figsize=(7, 6))  # Increased size

    # Heatmap with better formatting
    sns.heatmap(
        cov_matrix, 
        annot=True, 
        fmt=".1f",  # Reduce decimal places
        cmap="coolwarm", 
        center=0, 
        linewidths=0.5, 
        square=True, 
        cbar_kws={"shrink": 0.75},  # Adjust color bar size
        annot_kws={"size": 8},  # Reduce annotation font size
    )

    # Adjust tick labels
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    # Add a title with padding
    plt.title(f"{colour} Covariance Matrix Heatmap -- Normalised Features", fontsize=14, fontweight="bold", pad=20)

    # Adjust layout to fit everything
    plt.tight_layout()

    # Show plot
    plt.savefig(f"./plots/{colour}_cov_plot.png")
    plt.show()

def min_max_mean_std(df):
    stats = {}
    for col in df.columns:
        stats[col] = {
            'min': round(df[col].min(), 2),
            'max': round(df[col].max(), 2),
            'mean': round(df[col].mean(), 2),
            'std': round(df[col].std(), 2)
        }
    return stats

def class_balance(df, colour):
    print(df["quality"].value_counts())
    sns.barplot(x=df["quality"].value_counts().index,
            y=df["quality"].value_counts().values,
            palette="viridis")
    
    plt.xlabel("Quality")
    plt.ylabel("Count")
    plt.title(f"Quality Class Distribution - {colour} Wine")
    plt.savefig(f"./plots/{colour}_quality_dist_plot.png")
    plt.show()

def plot_feature(df, feature, colour):
    plt.scatter(df.index, df[feature])
    plt.xlabel("Index")
    plt.ylabel(f"Distribution of {feature} feature")
    plt.title(f"Distribution of {feature} - {colour}")
    plt.savefig(f"./plots/{feature}_{colour}_dist_plot.png")
    plt.show()

red_df = pd.read_csv("winequality-red.csv", sep=';')
white_df = pd.read_csv("winequality-white.csv", sep=';')

plot_feature(red_df, "total sulfur dioxide", "Red")
plot_feature(white_df, "total sulfur dioxide", "White")

