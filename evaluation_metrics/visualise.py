# visualize the trajectory
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os
from utils import renormlizer
import pickle

def visualise_gan(data_continuous_real, data_continuous_syn, data_discrete_real, data_discrete_syn, inx, num_dim=12, num_plot=10, SAVE_PATH="logs/"):

    # renorm
    DATA_PATH = "data/real/mimic/"
    min_val_con = np.load(os.path.join(DATA_PATH, "norm_stats.npz"))["min_val"]
    max_val_con = np.load(os.path.join(DATA_PATH, "norm_stats.npz"))["max_val"]
    data_continuous_real = renormlizer(data_continuous_real, max_val_con, min_val_con)
    data_continuous_syn = renormlizer(data_continuous_syn, max_val_con, min_val_con)

    sns.set_style("whitegrid")

    fig, axes = plt.subplots(4, num_dim, figsize=(100, 50)) 

    # Set the ticks and ticklabels for all axes
    plt.setp(axes, xticks=[0, 3, 6, 9, 12, 15, 18, 21, 24])

    # randomly select [num_plot, num_dim] features from real data and visualise
    c_dim_list  = random.sample(list(range(data_continuous_real.shape[2])), num_dim)
    c_pid_index = random.sample(list(range(len(data_continuous_real))), num_plot)
    
    for i in range(num_dim):
        df = pd.DataFrame(data_continuous_real[c_pid_index, :, c_dim_list[i]])
        sns.lineplot(ax=axes[0, i], data=df.T, palette=sns.color_palette('Greens', n_colors=num_plot))

    # randomly select [num_plot, num_dim] features from synthetic data and visualise
    c_pid_index = random.sample(list(range(len(data_continuous_syn))), num_plot)
    
    for i in range(num_dim):
        df = pd.DataFrame(data_continuous_syn[c_pid_index, :, c_dim_list[i]])
        sns.lineplot(ax=axes[1, i], data=df.T, palette=sns.color_palette('Reds', n_colors=num_plot))

    # randomly select [num_plot, num_dim] features from real data and visualise
    d_dim_list = random.sample(list(range(data_discrete_real.shape[2])), num_dim)
    d_pid_index = random.sample(list(range(len(data_discrete_real))), num_plot)

    for i in range(num_dim):
        df = pd.DataFrame(data_discrete_real[d_pid_index, :, d_dim_list[i]])
        sns.lineplot(ax=axes[2, i], data=df.T, palette=sns.color_palette('Greens', n_colors=num_plot))

    # randomly select [num_plot, num_dim] features from fake data and visualise
    d_pid_index = random.sample(list(range(len(data_discrete_syn))), num_plot)

    for i in range(num_dim):
        df = pd.DataFrame(data_discrete_syn[d_pid_index, :, d_dim_list[i]])
        sns.lineplot(ax=axes[3, i], data=df.T, palette=sns.color_palette('Reds', n_colors=num_plot))

    fig.savefig(os.path.join(SAVE_PATH, 'visualise_gan_epoch_' + str(inx) + '.pdf'), format='pdf')

def visualise_vae(data_continuous_real, data_continuous_syn, data_discrete_real, data_discrete_syn, inx, num_dim=8, num_plot=10, SAVE_PATH="logs/"):

    # renorm
    DATA_PATH = "data/real/mimic/"
    min_val_con = np.load(os.path.join(DATA_PATH, "norm_stats.npz"))["min_val"]
    max_val_con = np.load(os.path.join(DATA_PATH, "norm_stats.npz"))["max_val"]
    data_continuous_real = renormlizer(data_continuous_real, max_val_con, min_val_con)
    data_continuous_syn = renormlizer(data_continuous_syn, max_val_con, min_val_con)

    sns.set_style("whitegrid")

    fig, axes = plt.subplots(2, num_dim, figsize=(100, 50))

    # Set the ticks and ticklabels for all axes
    plt.setp(axes, xticks=[0, 3, 6, 9, 12, 15, 18, 21, 24])

    c_dim_list  = random.sample(list(range(data_continuous_real.shape[2])), num_dim)
    c_pid_index = random.sample(list(range(len(data_continuous_syn))), num_plot)
    
    for i in range(len(c_dim_list)):
        df = pd.DataFrame(data_continuous_real[c_pid_index, :, c_dim_list[i]])
        sns.lineplot(ax=axes[0, i], data=df.T, palette=sns.color_palette('Greens', n_colors=num_plot))
    
    for i in range(len(c_dim_list)):
        df = pd.DataFrame(data_continuous_syn[c_pid_index, :, c_dim_list[i]])
        sns.lineplot(ax=axes[0, i], data=df.T, marker='o', palette=sns.color_palette('Reds', n_colors=num_plot))

    d_dim_list = random.sample(list(range(data_discrete_real.shape[2])), num_dim) 
    d_pid_index = random.sample(list(range(len(data_discrete_syn))), num_plot)

    for i in range(len(d_dim_list)):
        df = pd.DataFrame(data_discrete_real[d_pid_index, :, d_dim_list[i]])
        sns.lineplot(ax=axes[1, i], data=df.T, palette=sns.color_palette('Greens', n_colors=num_plot))

    for i in range(len(d_dim_list)):
        df = pd.DataFrame(data_discrete_syn[d_pid_index, :, d_dim_list[i]])
        sns.lineplot(ax=axes[1, i], data=df.T, marker='o', palette=sns.color_palette('Reds', n_colors=num_plot))

    fig.savefig(os.path.join(SAVE_PATH, 'visualise_vae_epoch_' + str(inx) + '.pdf'), format='pdf')
