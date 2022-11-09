# Synthesizing mixed-type longitudinal EHR data with EHR-M-GAN

In this work, we propose a generative adversarial network (GAN) entitled EHR-M-GAN which simultaneously synthesizes mixed-type timeseries EHR data (e.g., continuous-valued timeseries and discrete-valued timeseries). EHR-M-GAN is capable of capturing the multidimensional, heterogeneous, and correlated temporal dynamics in patient trajectories.

This repository contains a Tensorflow implementation of EHR-M-GAN. For details, please see **Generating Synthetic Mixed-type Longitudinal Electronic Health Records for Artificial Intelligent Applications**.
[[Arxiv paper link](https://arxiv.org/abs/2112.12047)]


# Requirements

The code requires

* Python 3.6 or higher
* Tensorflow 1.14.0 or higher
* Numpy
* Sklearn
* Pickle
* Matplotlib
* Seaborn
* Pandas

# Datasets

## Download

All datasets are publicly available from PhysioNet, and can be downloaded from the following links:

1) [MIMIC-III (v1.4)](https://physionet.org/content/mimiciii/1.4/)
2) [eICU-CRD (v2.0)](https://physionet.org/content/eicu-crd/2.0/)
3) [HiRID (v1.1.1)](https://physionet.org/content/hirid/1.1.1/)

## Pre-processing

In order to preprocess the datasets for running EHR-M-GAN, please refer to the following [repository](https://github.com/jli0117/preprocessing_physionet).

# Training

## Code explanation

- `main_train.py` : Use mixed-type timeseries EHR data as training set to generate synthetic data
- `networks.py` : Components (generators and discriminators in the sequentially coupled GANs, encoders and decoders in the dual-VAE) in the model
- `m3gan.py` : Pretrain the latent representations and optimize the adversarial learning networks
- `Constrastivelosslayer.py` : The contrastive loss function in learning the shared VAE representations
- `Bilateral_lstm_cell.py` : The proposed Bilateral LSTM cell (single-layer)
- `Bilateral_lstm_class.py` : The proposed Bilateral LSTM network with multiple layers
- `init_state.py` : Initial state function for recurrent neural networks
- `utils.py` : other utility fucntions for adversarial training

## Running the code
To train the model(s) in the paper, simply run this command:

```
python main_train.py --dataset mimic --num_pre_epochs 500 --num_epochs 800 --epoch_ckpt_freq 100 
```

For training the conditional extension of EHR-M-GAN in the paper, run this command:
```
python main_train.py --dataset mimic --conditional True --num_labels 1 --num_pre_epochs 500 --num_epochs 800 --epoch_ckpt_freq 100 
```

