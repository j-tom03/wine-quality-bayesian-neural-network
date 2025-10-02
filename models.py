from preprocessing import preprocess
from plots import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import time

from sklearn.model_selection import train_test_split
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from tqdm.auto import trange
import torch

class BNNRegression(PyroModule):
    def __init__(self, in_dim=11, out_dim=1, hid_dim=8, prior_scale=10.):
        super().__init__()

        self.activation = nn.Tanh()
        self.layer1 = PyroModule[nn.Linear](in_dim, hid_dim)
        self.layer2 = PyroModule[nn.Linear](hid_dim, hid_dim)
        self.layer3 = PyroModule[nn.Linear](hid_dim, out_dim)

        self.layer1.weight = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim, in_dim]).to_event(2))
        self.layer1.bias = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim]).to_event(1))
        self.layer2.weight = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim, hid_dim]).to_event(2))
        self.layer2.bias = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim]).to_event(1))
        self.layer3.weight = PyroSample(dist.Normal(0., prior_scale).expand([out_dim, hid_dim]).to_event(2))
        self.layer3.bias = PyroSample(dist.Normal(0., prior_scale).expand([out_dim]).to_event(1))

    def forward(self, x, y=None):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x = self.activation(self.layer3(x2))
        mu = self.layer3(x2).squeeze()
        sigma = pyro.sample("sigma", dist.Gamma(2.0, 1.0))

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma * sigma), obs=y)
        return mu
    
class BNNClassification(PyroModule):
    def __init__(self, in_dim=11, out_dim=10, hid_dim=8, prior_scale=1.):
        super().__init__()

        self.layer1 = PyroModule[nn.Linear](in_dim, hid_dim)
        self.layer2 = PyroModule[nn.Linear](hid_dim, hid_dim)
        self.layer3 = PyroModule[nn.Linear](hid_dim, out_dim)

        self.layer1.weight = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim, in_dim]).to_event(2))
        self.layer1.bias = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim]).to_event(1))
        self.layer2.weight = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim, hid_dim]).to_event(2))
        self.layer2.bias = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim]).to_event(1))
        self.layer3.weight = PyroSample(dist.Normal(0., prior_scale).expand([out_dim, hid_dim]).to_event(2))
        self.layer3.bias = PyroSample(dist.Normal(0., prior_scale).expand([out_dim]).to_event(1))

    def forward(self, x, y=None):
        x = self.layer1(x)
        x = self.layer2(x)
        logits = self.layer3(x)
        logits = F.log_softmax(logits, dim=-1)

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits

def run_mcmc(df, test_size=0.3, bnn='Regression', colour='White'):
    start_time = time.time()

    pyro.clear_param_store()
    pyro.set_rng_seed(19)

    X, y = df.drop('quality', axis='columns').to_numpy(), df['quality'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=19)

    if bnn=='Regression':
        model = BNNRegression()

        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()
    elif bnn=='Classification':
        model = BNNClassification()

        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).long()
    else:
        return False

    nuts_kernel = NUTS(model, jit_compile=True)

    mcmc = MCMC(nuts_kernel, num_samples=100)

    mcmc.run(X_train, y_train)

    mcmc_samples = mcmc.get_samples()
    
    predictive = Predictive(model=model, posterior_samples=mcmc.get_samples())

    end_time = time.time()

    y_preds = predictive(X_test)
    samples_mcmc = predictive(X_test)

    if bnn=='Classification':
        y_preds = stats.mode(samples_mcmc['obs'].T.detach().numpy(), axis=1)[0].flatten()
        y_for_cm = y_preds
    else:
        y_preds = y_preds['obs'].T.detach().numpy().mean(axis=1)
        y_preds_rounded = list(map(round, y_preds))
        y_for_cm = y_preds_rounded

    y_test = y_test.numpy()
    y_train = y_train.numpy()

    mse, mae, acc, acc_within_1, acc_within_ci = evaluate_predictions(y_preds, samples_mcmc, y_test)

    results_dict = {
        "method": ["MCMC"],
        "reg/class": [bnn],
        "colour": [colour],
        "time": [round((end_time - start_time), 2)],
        "mse": [mse],
        "mae": [mae],
        "acc": [acc],
        "acc_within_1": [acc_within_1],
        "acc_within_ci": [acc_within_ci] 
    }

    results_to_csv_file(results_dict)

    plot_confusion(y_for_cm, y_test, bnn=bnn, colour=colour, method='MCMC', savefig=True)

    return predictive, samples_mcmc, mcmc_samples

def run_vi(df, test_size=0.3, bnn='Regression', colour='White'):
    start_time = time.time()

    pyro.clear_param_store()
    pyro.set_rng_seed(19)

    X, y = df.drop('quality', axis='columns').to_numpy(), df['quality'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=19)

    if bnn=='Regression':
        model = BNNRegression()

        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()
    elif bnn=='Classification':
        model = BNNClassification()

        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).long()
    else:
        return False

    mean_field_guide = AutoDiagonalNormal(model)
    optimizer = pyro.optim.Adam({"lr": 0.01})

    svi = SVI(model, mean_field_guide, optimizer, loss=Trace_ELBO())

    num_epochs = 30000
    progress_bar = trange(num_epochs)

    for epoch in progress_bar:
        loss = svi.step(X_train, y_train)
        progress_bar.set_postfix(loss=f"{loss / X_train.shape[0]:.3f}")

    predictive = Predictive(model, guide=mean_field_guide, num_samples=500)

    end_time = time.time()

    y_preds = predictive(X_test)
    samples_vi = predictive(X_test)

    if bnn=='Classification':
        y_preds = stats.mode(samples_vi['obs'].T.detach().numpy(), axis=1)[0].flatten()
        y_for_cm = y_preds
    else:
        y_preds = y_preds['obs'].T.detach().numpy().mean(axis=1)
        y_preds_rounded = list(map(round, y_preds))
        y_for_cm = y_preds_rounded

    y_test = y_test.numpy()
    y_train = y_train.numpy()

    mse, mae, acc, acc_within_1, acc_within_ci = evaluate_predictions(y_preds, samples_vi, y_test)

    results_dict = {
        "method": ["VI"],
        "reg/class": [bnn],
        "colour": [colour],
        "time": [round((end_time - start_time), 2)],
        "mse": [mse],
        "mae": [mae],
        "acc": [acc],
        "acc_within_1": [acc_within_1],
        "acc_within_ci": [acc_within_ci] 
    }

    results_to_csv_file(results_dict)

    plot_confusion(y_for_cm, y_test, bnn=bnn, colour=colour, method='VI', savefig=True)

    return predictive, samples_vi, mean_field_guide

def evaluate_predictions(y_preds, samples, y_test, interval_size=0.95):
    y_preds_rounded = list(map(round, y_preds))
    tol = (1-interval_size)/2
    lower_bounds = np.quantile(samples['obs'].detach().numpy(), tol, axis=0)
    upper_bounds = np.quantile(samples['obs'].detach().numpy(), 1-tol, axis=0)
    
    sae = 0
    sse = 0
    correct = 0
    sort_of_correct = 0
    within_ci = 0
    n = len(y_preds)
    
    for i in range(n):
        if y_test[i] == y_preds_rounded[i]:
            correct += 1
        if y_preds_rounded[i] - 1 <= y_test[i] <= y_preds_rounded[i] + 1:
            sort_of_correct += 1
        if lower_bounds[i] <= y_test[i] <= upper_bounds[i]:
            within_ci += 1
        
        sae += abs(y_test[i] - y_preds[i])
        sse += (y_test[i] - y_preds[i]) ** 2
    
    mae = sae / n
    mse = sse / n
    
    print(f"MSE: {mse} MAE: {mae}")
    print(f"Correct: {correct} Accuracy: {correct / n}")
    print(f"+-1: {sort_of_correct} Accuracy: {sort_of_correct / n}")
    print(f"Within Confidence Interval: {within_ci} Accuracy: {within_ci / n}")

    return mse, mae, correct/n, sort_of_correct/n, within_ci/n

def reset_csv(filename="results.csv"):
    f = open(filename, mode="w")
    f.write("method,reg/class,colour,time,mse,mae,acc,acc_within_1,acc_within_ci")
    f.close()

def results_to_csv_file(info_dict, filename='results.csv'):
    df = pd.DataFrame(info_dict)
    df.to_csv(filename, index=False, mode='a', header=False)

if __name__=="__main__":
    reset_csv()
    red_df = pd.read_csv("./data/winequality-red.csv", sep=';')
    white_df = pd.read_csv("./data/winequality-white.csv", sep=';')

    red_df = preprocess(red_df, "Red")
    white_df = preprocess(white_df, "White")

    methods = ['Regression', 'Classification']
    colour_strings = ['Red', 'White']
    dfs = [red_df, white_df]

    combinations = [
        (red_df, 'Regression', 'Red'),
        (red_df, 'Classification', 'Red'),
        (white_df, 'Regression', 'White'),
        (white_df, 'Classification', 'White')        
        ]

    for df, method, colour in combinations:
        mcmc_predictive, samples_mcmc, mcmc_samples = run_mcmc(df, bnn=method, colour=colour)
        vi_predictive, samples_vi, mean_field_guide_vi = run_vi(df, bnn=method, colour=colour)

        plot_uncertainty(samples_vi, samples_mcmc, bnn=method, colour=colour, savefig=True)
        plot_uncertainty_smooth(samples_vi, samples_mcmc, bnn=method, colour=colour, savefig=True)
        plot_parameter_distributions(mcmc_samples, mean_field_guide_vi, param_name="layer1.weight", bnn=method, colour=colour, savefig=True)
        plot_posterior_distributions(samples_vi, samples_mcmc, bnn=method, colour=colour, savefig=True)
