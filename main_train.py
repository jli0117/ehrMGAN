import tensorflow as tf
import numpy as np
import pickle
import os
from networks import C_VAE_NET, D_VAE_NET, C_GAN_NET, D_GAN_NET
from m3gan import m3gan

import timeit
start = timeit.default_timer()

data_path = 'data/real/mimic/'

with open(os.path.join(data_path, 'vital_sign_24hrs.pkl'), 'rb') as f:
    vital_labs_3D = pickle.load(f)

with open(os.path.join(data_path, 'med_interv_24hrs.pkl'), 'rb') as f:
    medical_interv_3D = pickle.load(f)

with open(os.path.join(data_path, 'statics_label_2.pkl'), 'rb') as f:
    statics_label = pickle.load(f)

statics_label = np.asarray(statics_label)[:, 0].reshape([-1, 1])
continuous_x = vital_labs_3D
discrete_x = medical_interv_3D

# params:
batch_size = 256
time_steps = continuous_x.shape[1]
num_pre_epochs = 5000
num_epochs = 8000
epoch_ckpt_freq = 100 

"""
_dim: dimensionality of features
_z_size: dimensionality of latent space in VAE
_noise_dim: dimensionality of input in G
"""
shared_latent_dim = 25

c_dim = continuous_x.shape[2]
c_z_size = shared_latent_dim
c_noise_dim = int(c_dim/2)

d_dim = discrete_x.shape[2]
d_z_size = shared_latent_dim
d_noise_dim = int(d_dim/2)

conditional = True
num_labels = 1

# networks for continuousGAN
c_vae = C_VAE_NET(batch_size=batch_size, time_steps=time_steps, dim=c_dim, z_dim=c_z_size,
                  conditional=conditional, num_labels=num_labels)

c_gan = C_GAN_NET(batch_size=batch_size, noise_dim=c_noise_dim, dim=c_dim,
                  gen_dim=c_z_size, time_steps=time_steps,
                  conditional=conditional, num_labels=num_labels)

# networks for discreteGAN
d_vae = D_VAE_NET(batch_size=batch_size, time_steps=time_steps, dim=d_dim, z_dim=d_z_size,
                  conditional=conditional, num_labels=num_labels)

d_gan = D_GAN_NET(batch_size=batch_size, noise_dim=d_noise_dim, dim=d_dim,
                  gen_dim=d_z_size, time_steps=time_steps,
                  conditional=conditional, num_labels=num_labels)

# create data directory for saving
checkpoint_dir = "data/checkpoint/"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

tf.reset_default_graph()
run_config = tf.ConfigProto()
with tf.Session(config=run_config) as sess:
    model = m3gan(sess=sess,
                  batch_size=batch_size,
                  time_steps=time_steps,
                  num_pre_epochs=num_pre_epochs,
                  num_epochs=num_epochs,
                  checkpoint_dir=checkpoint_dir,
                  epoch_ckpt_freq=epoch_ckpt_freq,
                  c_dim=c_dim, c_noise_dim=c_noise_dim,
                  c_z_size=c_z_size, c_data_sample=continuous_x,
                  c_vae=c_vae, c_gan=c_gan,
                  d_dim=d_dim, d_noise_dim=d_noise_dim,
                  d_z_size=d_z_size, d_data_sample=discrete_x,
                  d_vae=d_vae, d_gan=d_gan,
                  conditional=conditional, num_labels=num_labels,
                  statics_label=statics_label)
    model.build()
    model.train()

stop = timeit.default_timer()
print('Time: ', stop - start)


