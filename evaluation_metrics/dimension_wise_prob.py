import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

DATA_SOURCE = "mimic"

def plot_sub(real_prob, fake_prob, feature, ax, name, cc, rmse):
    df = pd.DataFrame({'real': real_prob,  'fake': fake_prob, "feature": feature})
    sns.scatterplot(ax=ax, data=df, x='real', y='fake', hue="feature", s=10, alpha=0.8, edgecolor='none', legend=None, palette='Paired_r')
    sns.lineplot(ax=ax, x=[0, 1], y=[0, 1], linewidth=0.5, color="darkgrey")
    ax.set_title(name, fontsize=11)
    ax.set(xlabel="Bernoulli success probability of real data")
    ax.set(ylabel="Bernoulli success probability of synthetic data")
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.text(0.75, 0.05, 'CC='+str(cc), fontsize=9)
    ax.text(0.75, 0.05, 'RMSE='+str(rmse), fontsize=9)

def cal_cc(real_prob, fake_prob):
    return float("{:.4f}".format(np.corrcoef(real_prob, fake_prob)[0, 1]))

def cal_rmse(real_prob, fake_prob):
    return float("{:.4f}".format(sqrt(mean_squared_error(real_prob, fake_prob)))) 
    

sns.set_style("whitegrid", {'grid.linestyle': ' '})
fig, ax = plt.subplots(figsize=(4.2, 3.8))

real_prob = np.mean(discrete_x_real, axis=0)
fake_prob = np.mean(discrete_x_fake, axis=0)
feature = np.concatenate([([i]* discrete_x.shape[1] ) for i in list(range(discrete_x.shape[-1])) ], axis=0)
cc_value = cal_cc(real_prob, fake_prob)
rmse_value = cal_rmse(real_prob, fake_prob)
plot_sub(real_prob, fake_prob, feature, ax, name="GAN", cc=cc_value, rmse=rmse_value)
fig.show()
