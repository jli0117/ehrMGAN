import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def cross_pearson_plot(c_x_real, c_x_syn, d_x_real, d_x_syn):
    plt.rcParams["figure.figsize"] = (20, 10)

    sns.set(font="Ubuntu", font_scale=0.5)
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 2)

    def single_plot(c_data, d_data, timepoints, index):
        timepoints = [0, 5, 11, 17, 23, 29, 35, 41, 47]

        c_names = ["SpO2", "SBP", "RR", "HR", "Temp"]
        d_names = ["Vent.", "Vaso."]
        c_feature_names = []
        c_data_list = []
        dim_c = [61, 91, 87, 38, 92]
        
        for i in range(len(dim_c)):
            for j in timepoints:
                t_ = j
                c_data_list.append(c_data[:, t_, dim_c[i]])
                c_feature_names.append(c_names[i] + "_" + str(t_+1).zfill(2))
        c_data_ = np.array(c_data_list)
        c_data_ = np.transpose(c_data_, [1, 0])

        _, time_steps, dim = d_data.shape
        dim_d = [0, 1]
        d_feature_names = []
        d_data_list = []
        for i in range(len(dim_d)):
            for j in timepoints:
                t_ = j
                d_data_list.append(d_data[:, t_, dim_d[i]])
                d_feature_names.append(d_names[i] + "_" + str(t_+1).zfill(2))
        d_data_ = np.array(d_data_list)
        d_data_ = np.transpose(d_data_, [1, 0])

        d = pd.DataFrame(data=np.concatenate((c_data_, d_data_), axis=1), columns=c_feature_names+d_feature_names)
        corr = d.corr()

        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, ax=axes[index], mask=mask, cmap="YlGnBu", vmin=-1., vmax=+1., center=0,
                    square=True, cbar_kws={"shrink": .5})
        axes[index].set_xticklabels(axes[index].get_xticklabels(), rotation=75)

    single_plot(c_x_real, d_x_real, 0)  
    single_plot(c_x_syn, d_x_syn, 1)   

    fig.show()
    fig.savefig('res/pearson_corr.pdf', format='pdf')