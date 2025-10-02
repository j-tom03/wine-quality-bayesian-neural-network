import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_confusion(y_pred, y_test, bnn='Regression', colour='White', method='MCMC', savefig=False):
    y_pred_rounded = np.round(y_pred)
    y_pred_rounded = y_pred_rounded.astype(np.int64)
    y_test = y_test.astype(np.int64)

    cm = confusion_matrix(y_pred_rounded, y_test)

    min_lab = min(min(y_pred_rounded), np.min(y_test))
    max_lab = max(max(y_pred_rounded), np.max(y_test))
    labels = [x for x in range(min_lab, max_lab+1)]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix -- {method} {bnn} -- {colour} Dataset")
    plt.tight_layout()
    if savefig:
        plt.savefig(f"./plots/results/confusion/confusion_{bnn}_{method}_{colour}.png", dpi=300)
        plt.close()
    else:
        plt.show()

def plot_posterior_distributions(samples_vi, samples_mcmc, bnn='Regression', colour='White', savefig=False):
    vi_samples = samples_vi["obs"].detach().numpy().flatten()
    mcmc_samples = samples_mcmc["obs"].detach().numpy().flatten()
    
    plt.figure(figsize=(8, 6))
    sns.histplot(vi_samples, kde=True, label="VI", color='blue', bins=30, stat='density', alpha=0.6)
    sns.histplot(mcmc_samples, kde=True, label="MCMC", color='red', bins=30, stat='density', alpha=0.6)
    
    plt.xlabel("Predicted Output")
    plt.ylabel("Density")
    plt.title(f"Posterior Predictive Distribution -- {bnn} {colour} Dataset")
    plt.legend()
    if savefig:
        plt.savefig(f"./plots/results/posterior_dist/posterior_dist_{bnn}_{colour}.png", dpi=300)
        plt.close()
    else:
        plt.show()

def plot_parameter_distributions(mcmc_samples, vi_guide, param_name="layer1.weight", bnn='Regression', colour='White', savefig=False):
    param_mcmc = mcmc_samples[param_name].detach().numpy().flatten()
    param_vi = vi_guide.median()[param_name].detach().numpy().flatten()
    
    plt.figure(figsize=(8, 6))
    sns.histplot(param_mcmc, kde=True, label="MCMC", color='red', bins=30, stat='density', alpha=0.6)
    sns.histplot(param_vi, kde=True, label="VI (Mean Field Approx.)", color='blue', bins=30, stat='density', alpha=0.6)
    
    plt.xlabel("Parameter Value")
    plt.ylabel("Density")
    plt.title(f"Posterior Distribution of {param_name} -- {bnn} {colour} Dataset")
    plt.legend()
    if savefig:
        plt.savefig(f"./plots/results/parameter_dist/parameter_dist_{bnn}_{colour}.png" , dpi=300)
        plt.close()
    else:
        plt.show()

def plot_uncertainty(samples_vi, samples_mcmc, bnn='Regression', colour='White', savefig=False):
    std_vi = samples_vi["obs"].detach().numpy().std(axis=0)
    std_mcmc = samples_mcmc["obs"].detach().numpy().std(axis=0)
    
    plt.figure(figsize=(8, 6))
    plt.plot(std_vi, label="VI Uncertainty", color='blue')
    plt.plot(std_mcmc, label="MCMC Uncertainty", color='red')
    
    plt.xlabel("Test Sample Index")
    plt.ylabel("Standard Deviation of Predictions")
    plt.title(f"Uncertainty Comparison -- {bnn} {colour} Dataset")
    plt.legend()
    if savefig:
        plt.savefig(f"./plots/results/uncertainty/uncertainty_{bnn}_{colour}.png", dpi=300)
        plt.close()
    else:
        plt.show()

def plot_uncertainty_smooth(samples_vi, samples_mcmc, bnn='Regression', colour='White', savefig=False):
    std_vi = samples_vi["obs"].detach().numpy().std(axis=0)
    std_mcmc = samples_mcmc["obs"].detach().numpy().std(axis=0)

    std_vi = np.convolve(std_vi.astype(float), np.ones(20)/20, mode='same')
    std_mcmc = np.convolve(std_mcmc.astype(float), np.ones(20)/20, mode='same')
    
    plt.figure(figsize=(8, 6))
    plt.plot(std_vi, label="VI Uncertainty", color='blue')
    plt.plot(std_mcmc, label="MCMC Uncertainty", color='red')
    
    plt.xlabel("Test Sample Index")
    plt.ylabel("Standard Deviation of Predictions")
    plt.title(f"Smoothed Uncertainty Comparison -- {bnn} {colour} Dataset")
    plt.legend()
    if savefig:
        plt.savefig(f"./plots/results/uncertainty_smooth/uncertainty_smoothed_{bnn}_{colour}.png", dpi=300)
        plt.close()
    else:
        plt.show()