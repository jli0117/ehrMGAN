import tensorflow as tf
import numpy as np
import os
from Contrastivelosslayer import nt_xent_loss
from utils import ones_target, zeros_target

class m3gan(object):
    def __init__(self, sess,
                 # -- shared params:
                 batch_size, time_steps,
                 num_pre_epochs, num_epochs,
                 checkpoint_dir, epoch_ckpt_freq,
                 # -- params for continuous-GAN
                 c_dim, c_noise_dim,
                 c_z_size, c_data_sample,
                 c_gan, c_vae,
                 # -- params for discrete-GAN
                 d_dim, d_noise_dim,
                 d_z_size, d_data_sample,
                 d_gan, d_vae,
                 # -- params for training
                 d_rounds=1, g_rounds=3, v_rounds=1,
                 # -- label information
                 conditional=False, num_labels=0,
                 statics_label=None):

        self.sess = sess
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.num_pre_epochs = num_pre_epochs
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.epoch_ckpt_freq = epoch_ckpt_freq
        self.statics_label = statics_label

        # params for continuous-GAN
        self.c_dim = c_dim
        self.c_noise_dim = c_noise_dim
        self.c_z_size = c_z_size
        self.c_data_sample = c_data_sample
        self.c_rnn_vae_net = c_vae
        self.cgan = c_gan

        # params for discrete-GAN
        self.d_dim = d_dim
        self.d_noise_dim = d_noise_dim
        self.d_z_size = d_z_size
        self.d_data_sample = d_data_sample
        self.d_rnn_vae_net = d_vae
        self.dgan = d_gan

        # params for training
        self.d_rounds = d_rounds
        self.g_rounds = g_rounds
        self.v_rounds = v_rounds

        # params for label information
        self.num_labels = num_labels
        self.conditional = conditional

    def build(self):
        self.build_tf_graph()
        self.build_loss()
        self.build_summary()
        self.saver = tf.train.Saver()

    def save(self, global_id, model_name=None, checkpoint_dir=None):
        self.saver.save(self.sess, os.path.join(
            checkpoint_dir, model_name), global_step=global_id)

    def load(self, model_name=None, checkpoint_dir=None):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        global_id = int(ckpt_name[len(model_name) + 1:])
        return global_id

    def embed_vis(self, embedding_var, type):
        saving_dir = 'logs/tf_summary'

        # if type == "raw_data":
        #     saving_dir = 'logs/raw_data_emb'
        #     if not os.path.exists(saving_dir):
        #         os.makedirs(saving_dir)
        # elif type == "latent_space":
        #     saving_dir = 'logs/latent_space_emb'
        #     if not os.path.exists(saving_dir):
        #         os.makedirs(saving_dir)

        metadata_file = os.path.join(saving_dir, 'metadata_classes.tsv')
        with open(metadata_file, 'w') as f:
            f.write("patient_status\tpatient_status1\n")
            for i in range(embedding_var.shape[0]):
                c1 = self.statics_label[i, 0]   # patient status
                f.write("%s\t%s\n" % (c1, c1))
        f.close()

        """Setup for Tensorboard embedding visualization"""
        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = embedding_var.name
        embed.metadata_path = os.path.join('', 'metadata_classes.tsv')
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(self.summary_writer, config)
        saver_images = tf.train.Saver([embedding_var])
        saver_images.save(self.sess, os.path.join(saving_dir, 'embeddings.ckpt'))

    def build_tf_graph(self):
        # Step 1: VAE pretraining 
        if self.conditional:
            self.real_data_label_pl = tf.placeholder(
                dtype=float, shape=[self.batch_size, self.num_labels], name="real_data_label")

        self.c_real_data_pl = tf.placeholder(
            dtype=float, shape=[self.batch_size, self.time_steps, self.c_dim], name="continuous_real_data")

        if self.conditional:
            self.c_decoded_output, self.c_vae_sigma, self.c_vae_mu, self.c_vae_logsigma, self.c_enc_z = \
                self.c_rnn_vae_net.build_vae(self.c_real_data_pl, self.real_data_label_pl)
        else:
            self.c_decoded_output, self.c_vae_sigma, self.c_vae_mu, self.c_vae_logsigma, self.c_enc_z = \
                self.c_rnn_vae_net.build_vae(self.c_real_data_pl)

        self.c_vae_test_data_pl = tf.placeholder(
            dtype=float, shape=[self.batch_size, self.time_steps, self.c_dim], name="vae_validation_c_data")
        if self.conditional:
            self.c_vae_test_decoded, _, _, _, _ = \
                self.c_rnn_vae_net.build_vae(self.c_vae_test_data_pl, self.real_data_label_pl)
        else:
            self.c_vae_test_decoded, _, _, _, _ = \
                self.c_rnn_vae_net.build_vae(self.c_vae_test_data_pl)

        self.d_real_data_pl = tf.placeholder(
            dtype=float, shape=[self.batch_size, self.time_steps, self.d_dim], name="discrete_real_data")
        if self.conditional:
            self.d_decoded_output, self.d_vae_sigma, self.d_vae_mu, self.d_vae_logsigma, self.d_enc_z = \
                self.d_rnn_vae_net.build_vae(self.d_real_data_pl, self.real_data_label_pl)
        else:
            self.d_decoded_output, self.d_vae_sigma, self.d_vae_mu, self.d_vae_logsigma, self.d_enc_z = \
                self.d_rnn_vae_net.build_vae(self.d_real_data_pl)

        self.d_vae_test_data_pl = tf.placeholder(
            dtype=float, shape=[self.batch_size, self.time_steps, self.d_dim], name="vae_validation_d_data")
        if self.conditional:
            self.d_vae_test_decoded, _, _, _, _ = \
                self.d_rnn_vae_net.build_vae(self.d_vae_test_data_pl, self.real_data_label_pl)
        else:
            self.d_vae_test_decoded, _, _, _, _ = \
                self.d_rnn_vae_net.build_vae(self.d_vae_test_data_pl)


        # Step 2: Generating
        self.c_gen_input_noise_pl = tf.placeholder(
            tf.float32, [None, self.time_steps, self.c_noise_dim], name="continuous_generator_input_noise")
        if self.conditional:
            c_initial_state = self.cgan.build_GenRNN(self.c_gen_input_noise_pl, self.real_data_label_pl)
        else:
            c_initial_state = self.cgan.build_GenRNN(self.c_gen_input_noise_pl)

        self.d_gen_input_noise_pl = tf.placeholder(
            tf.float32, [None, self.time_steps, self.d_noise_dim], name="discrete_generator_input_noise")
        if self.conditional:
            d_initial_state = self.dgan.build_GenRNN(self.d_gen_input_noise_pl, self.real_data_label_pl)
        else:
            d_initial_state = self.dgan.build_GenRNN(self.d_gen_input_noise_pl)

        ### sequentially coupled iteraing
        self.c_gen_output_latent = []
        self.d_gen_output_latent = []

        for t in range(self.time_steps):
            d_state = d_new_state if t > 0 else d_initial_state
            c_state = c_new_state if t > 0 else c_initial_state

            # _new_state is a tuple of (h_i, c_i)
            d_new_linear, d_new_state = self.dgan.gen_Onestep(t, [d_state, c_state])
            c_new_linear, c_new_state = self.cgan.gen_Onestep(t, [c_state, d_state])

            self.d_gen_output_latent.append(d_new_linear)
            self.c_gen_output_latent.append(c_new_linear)

        # Step3: Decoding
        if self.conditional:
            self.d_gen_decoded = self.d_rnn_vae_net.reconstruct_decoder(dec_input=self.d_gen_output_latent,
                                                                        conditions=self.real_data_label_pl)
        else:
            self.d_gen_decoded = self.d_rnn_vae_net.reconstruct_decoder(dec_input=self.d_gen_output_latent)
        self.d_gen_decoded = tf.unstack(self.d_gen_decoded, axis=1)

        if self.conditional:
            self.c_gen_decoded = self.c_rnn_vae_net.reconstruct_decoder(dec_input=self.c_gen_output_latent,
                                                                        conditions=self.real_data_label_pl)
        else:
            self.c_gen_decoded = self.c_rnn_vae_net.reconstruct_decoder(dec_input=self.c_gen_output_latent)
        self.c_gen_decoded = tf.unstack(self.c_gen_decoded, axis=1)

        # Step4: Discriminator

        self.c_fake, self.c_fake_prob, self.c_fake_fm = self.cgan.build_Discriminator(self.c_gen_decoded)
        self.c_real, self.c_real_prob, self.c_real_fm = self.cgan.build_Discriminator(tf.unstack(self.c_real_data_pl, axis=1))

        self.d_fake, self.d_fake_prob = self.dgan.build_Discriminator(self.d_gen_decoded)
        self.d_real, self.d_real_prob = self.dgan.build_Discriminator(tf.unstack(self.d_real_data_pl, axis=1))

    def build_loss(self):

        #################
        # (1) VAE loss  #
        #################
        alpha_re = 1
        alpha_kl = 0.5
        alpha_mt = 0.1
        alpha_ct = 0.1
        alpha_sm = 1

        x_latent_1 = tf.stack(self.c_enc_z, axis=1)
        x_latent_2 = tf.stack(self.d_enc_z, axis=1)
        self.vae_matching_loss = tf.losses.mean_squared_error(x_latent_1, x_latent_2)

        self.vae_contra_loss = nt_xent_loss(tf.reshape(x_latent_1, [x_latent_1.shape[0], -1]),
                                            tf.reshape(x_latent_2, [x_latent_2.shape[0], -1]), self.batch_size)

        if self.conditional:
            x_latent_1_ = x_latent_1[:, :, :-1]
            x_latent_2_ = x_latent_2[:, :, :-1]
            with tf.variable_scope("Shared_VAE/semantic_classifier"):
                vae_flatten_input = tf.compat.v1.layers.flatten(tf.concat([x_latent_1_, x_latent_2_], axis=-1))
                vae_hidden_layer = tf.layers.dense(vae_flatten_input, units=24, activation=tf.nn.relu)
                vae_logits = tf.layers.dense(vae_hidden_layer, units=4, activation=tf.nn.tanh)
            self.vae_semantics_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.squeeze(tf.cast(self.real_data_label_pl, dtype=tf.int32)), logits=vae_logits))

        self.c_re_loss = tf.losses.mean_squared_error(self.c_real_data_pl, self.c_decoded_output)
        c_kl_loss = [0] * self.time_steps 
        for t in range(self.time_steps):
            c_kl_loss[t] = 0.5 * (tf.reduce_sum(self.c_vae_sigma[t], 1) + tf.reduce_sum(
                tf.square(self.c_vae_mu[t]), 1) - tf.reduce_sum(self.c_vae_logsigma[t] + 1, 1))
        self.c_kl_loss = tf.reduce_mean(tf.add_n(c_kl_loss))

        if self.conditional:
            self.c_vae_loss = alpha_re * self.c_re_loss + \
                              alpha_kl * self.c_kl_loss + \
                              alpha_mt * self.vae_matching_loss + \
                              alpha_ct * self.vae_contra_loss + \
                              alpha_sm * self.vae_semantics_loss
        else:
            self.c_vae_loss = alpha_re * self.c_re_loss + \
                              alpha_kl * self.c_kl_loss + \
                              alpha_mt * self.vae_matching_loss + \
                              alpha_ct * self.vae_contra_loss

        self.c_vae_valid_loss = tf.losses.mean_squared_error(self.c_vae_test_data_pl, self.c_vae_test_decoded)

        self.d_re_loss = tf.losses.mean_squared_error(self.d_real_data_pl, self.d_decoded_output)
        d_kl_loss = [0] * self.time_steps
        for t in range(self.time_steps):
            d_kl_loss[t] = 0.5 * (tf.reduce_sum(self.d_vae_sigma[t], 1) + tf.reduce_sum(
                tf.square(self.d_vae_mu[t]), 1) - tf.reduce_sum(self.d_vae_logsigma[t] + 1, 1))
        self.d_kl_loss = 0.1 * tf.reduce_mean(tf.add_n(d_kl_loss))

        if self.conditional:
            self.d_vae_loss = alpha_re * self.d_re_loss + \
                              alpha_kl * self.d_kl_loss + \
                              alpha_mt * self.vae_matching_loss + \
                              alpha_ct * self.vae_contra_loss + \
                              alpha_sm * self.vae_semantics_loss
        else:
            self.d_vae_loss = alpha_re * self.d_re_loss + \
                              alpha_kl * self.d_kl_loss + \
                              alpha_mt * self.vae_matching_loss + \
                              alpha_ct * self.vae_contra_loss

        self.d_vae_valid_loss = tf.losses.mean_squared_error(self.d_vae_test_data_pl, self.d_vae_test_decoded)

        ###########################
        # (2) Discriminator loss  #
        ###########################

        self.continuous_d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.c_real, labels=ones_target(self.batch_size, min=0.7, max=1.2)))
        self.continuous_d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.c_fake, labels=zeros_target(self.batch_size, min=0.1, max=0.3)))
        self.continuous_d_loss = self.continuous_d_loss_real + self.continuous_d_loss_fake

        self.dicrete_d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.d_real, labels=ones_target(self.batch_size, min=0.8, max=0.9)))
        self.dicrete_d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.d_fake, labels=zeros_target(self.batch_size, min=0.1, max=0.1)))
        self.dicrete_d_loss = self.dicrete_d_loss_real + self.dicrete_d_loss_fake

        ###########################
        # (3) Generator loss      #
        ###########################
        self.c_gen_loss_adv = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.c_fake, labels=tf.ones_like(self.c_fake)))

        self.c_g_loss_v1 = tf.reduce_mean(
            tf.abs(tf.sqrt(tf.nn.moments(self.c_fake_fm, [0])[1] + 1e-6)
                   - tf.sqrt(tf.nn.moments(self.c_real_fm, [0])[1] + 1e-6)))

        self.c_g_loss_v2 = tf.reduce_mean(
            tf.abs(tf.sqrt(tf.abs(tf.nn.moments(self.c_fake_fm, [0])[0]))
                   - tf.sqrt(tf.abs(tf.nn.moments(self.c_real_fm, [0])[0]))))

        self.c_gen_loss_fm = self.c_g_loss_v1 + self.c_g_loss_v2

        c_beta_adv, c_beta_fm = 1, 20
        self.c_gen_loss = c_beta_adv * self.c_gen_loss_adv + c_beta_fm * self.c_gen_loss_fm

        self.d_gen_loss_adv = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.ones_like(self.d_fake)))

        self.d_g_loss_v1 = tf.reduce_mean(
            tf.abs(tf.sqrt(tf.nn.moments(tf.stack(self.d_gen_decoded, axis=1), [0])[1] + 1e-6) -
                   tf.sqrt(tf.nn.moments(self.d_real_data_pl, [0])[1] + 1e-6)))
        self.d_g_loss_v2 = tf.reduce_mean(
            tf.abs((tf.nn.moments(tf.stack(self.d_gen_decoded, axis=1), [0])[0]) -
                   (tf.nn.moments(self.d_real_data_pl, [0])[0])))
        self.d_gen_loss_fm = self.d_g_loss_v1 + self.d_g_loss_v2

        d_beta_adv, d_beta_fm = 1, 10
        self.d_gen_loss = d_beta_adv * self.d_gen_loss_adv + d_beta_fm * self.d_gen_loss_fm

        #######################
        # Optimizer           #
        #######################
        t_vars = tf.trainable_variables()
        c_vae_vars = [var for var in t_vars if 'Continuous_VAE' in var.name]
        d_vae_vars = [var for var in t_vars if 'Discrete_VAE' in var.name]
        s_vae_vars = [var for var in t_vars if 'Shared_VAE' in var.name]
        c_g_vars = [var for var in t_vars if 'Continuous_generator' in var.name]
        c_d_vars = [var for var in t_vars if 'Continuous_discriminator' in var.name]
        d_g_vars = [var for var in t_vars if 'Discrete_generator' in var.name]
        d_d_vars = [var for var in t_vars if 'Discrete_discriminator' in var.name]

        self.c_v_op_pre = tf.train.AdamOptimizer(learning_rate=0.05)\
            .minimize(self.c_vae_loss, var_list=c_vae_vars+s_vae_vars)

        self.d_v_op_pre = tf.train.AdamOptimizer(learning_rate=0.05)\
            .minimize(self.d_vae_loss, var_list=d_vae_vars+s_vae_vars)

        self.c_v_op = tf.train.AdamOptimizer(learning_rate=0.01) \
            .minimize(self.c_vae_loss, var_list=c_vae_vars + s_vae_vars)

        self.d_v_op = tf.train.AdamOptimizer(learning_rate=0.01) \
            .minimize(self.d_vae_loss, var_list=d_vae_vars + s_vae_vars)

        self.c_g_op = tf.train.AdamOptimizer(learning_rate=0.01)\
            .minimize(self.c_gen_loss, var_list=c_g_vars)

        self.d_g_op = tf.train.AdamOptimizer(learning_rate=0.01)\
            .minimize(self.d_gen_loss, var_list=d_g_vars)

        self.c_d_op = tf.train.AdamOptimizer(learning_rate=0.01)\
            .minimize(self.continuous_d_loss, var_list=c_d_vars)

        self.d_d_op = tf.train.AdamOptimizer(learning_rate=0.01)\
            .minimize(self.dicrete_d_loss, var_list=d_d_vars)

    def build_summary(self):

        self.c_vae_summary = []
        self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/reconstruction_loss", self.c_re_loss))
        self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/kl_divergence_loss", self.c_kl_loss))
        self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/matching_loss", self.vae_matching_loss))
        self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/contrastive_loss", self.vae_contra_loss))
        if self.conditional:
            self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/semantic_loss", self.vae_semantics_loss))
        self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/vae_loss", self.c_vae_loss))
        self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/validation_loss", self.c_vae_valid_loss))
        self.c_vae_summary = tf.summary.merge(self.c_vae_summary)

        self.d_vae_summary = []
        self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/reconstruction_loss", self.d_re_loss))
        self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/kl_divergence_loss", self.d_kl_loss))
        self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/matching_loss", self.vae_matching_loss))
        self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/contrastive_loss", self.vae_contra_loss))
        if self.conditional:
            self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/semantic_loss", self.vae_semantics_loss))
        self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/vae_loss", self.d_vae_loss))
        self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/validation_loss", self.d_vae_valid_loss))
        self.d_vae_summary = tf.summary.merge(self.d_vae_summary)

        self.c_discriminator_summary = []
        self.c_discriminator_summary.append(
            tf.summary.scalar("c_discriminator_loss/d_real", self.continuous_d_loss_real))
        self.c_discriminator_summary.append(
            tf.summary.scalar("c_discriminator_loss/d_fake", self.continuous_d_loss_fake))
        self.c_discriminator_summary.append(
            tf.summary.scalar("c_discriminator_loss/discriminator_loss", self.continuous_d_loss))
        self.c_discriminator_summary = tf.summary.merge(self.c_discriminator_summary)

        self.c_generator_summary = []
        self.c_generator_summary.append(
            tf.summary.scalar("c_generator_loss/adversarial_loss", self.c_gen_loss_adv))
        self.c_generator_summary.append(
            tf.summary.scalar("c_generator_loss/feature_matching_loss_v1", self.c_g_loss_v1))
        self.c_generator_summary.append(
            tf.summary.scalar("c_generator_loss/feature_matching_loss_v2", self.c_g_loss_v2))
        self.c_generator_summary.append(
            tf.summary.scalar("c_generator_loss/feature_matching_loss", self.c_gen_loss_fm))
        self.c_generator_summary.append(
            tf.summary.scalar("c_generator_loss/generator_loss", self.c_gen_loss))
        self.c_generator_summary = tf.summary.merge(self.c_generator_summary)

        self.d_discriminator_summary = []
        self.d_discriminator_summary.append(
            tf.summary.scalar("d_discriminator_loss/dicrete_d_loss_real", self.dicrete_d_loss_real))
        self.d_discriminator_summary.append(
            tf.summary.scalar("d_discriminator_loss/dicrete_d_loss_fake", self.dicrete_d_loss_fake))
        self.d_discriminator_summary.append(
            tf.summary.scalar("d_discriminator_loss/d_discriminator", self.dicrete_d_loss))
        self.d_discriminator_summary = tf.summary.merge(self.d_discriminator_summary)

        self.d_generator_summary = []
        self.d_generator_summary.append(
            tf.summary.scalar("d_generator_loss/g_loss_v1", self.d_g_loss_v1))
        self.d_generator_summary.append(
            tf.summary.scalar("d_generator_loss/g_loss_v2", self.d_g_loss_v2))
        self.d_generator_summary.append(
            tf.summary.scalar("d_generator_loss/d_gen_loss_fm", self.d_gen_loss_fm))
        self.d_generator_summary.append(
            tf.summary.scalar("d_generator_loss/d_gen_loss_adv", self.d_gen_loss_adv))
        self.d_generator_summary.append(
            tf.summary.scalar("d_generator_loss/d_generator", self.d_gen_loss))
        self.d_generator_summary = tf.summary.merge(self.d_generator_summary)


    def gen_input_noise(self, num_sample, T, noise_dim):
        return np.random.uniform(size=[num_sample, T, noise_dim])

    def train(self):
        self.summary_writer = tf.summary.FileWriter("logs/tf_summary", self.sess.graph)

        continuous_x = self.c_data_sample[: int(0.9 * self.c_data_sample.shape[0]), :, :]
        continuous_x_test = self.c_data_sample[int(0.9 * self.c_data_sample.shape[0]) : , :, :]

        discrete_x = self.d_data_sample[: int(0.9 * self.d_data_sample.shape[0]), :, :]
        discrete_x_test = self.d_data_sample[int(0.9 * self.d_data_sample.shape[0]):, :, :]

        if self.conditional:
            label_data = self.statics_label[: int(0.9 * self.d_data_sample.shape[0]), :]

        data_size = continuous_x.shape[0]
        num_batches = data_size // self.batch_size

        tf.global_variables_initializer().run()

        # pretrain step
        print('start pretraining')
        global_id = 0

        for pre in range(self.num_pre_epochs):

            random_idx = np.random.permutation(data_size)
            continuous_x_random = continuous_x[random_idx]
            discrete_x_random = discrete_x[random_idx]
            if self.conditional:
                label_data_random = label_data[random_idx]

            random_idx_ = np.random.permutation(continuous_x_test.shape[0])
            continuous_x_test_batch = continuous_x_test[random_idx_][:self.batch_size, :, :]
            discrete_x_test_batch = discrete_x_test[random_idx_][:self.batch_size, :, :]

            print("pretraining epoch %d" % pre)

            c_real_data_lst = []
            c_rec_data_lst = []
            d_real_data_lst = []
            d_rec_data_lst = []

            for b in range(num_batches):

                feed_dict = {}
                feed_dict[self.c_real_data_pl] = continuous_x_random[b * self.batch_size: (b + 1) * self.batch_size]
                feed_dict[self.c_vae_test_data_pl] = continuous_x_test_batch
                feed_dict[self.d_real_data_pl] = discrete_x_random[b * self.batch_size: (b + 1) * self.batch_size]
                feed_dict[self.d_vae_test_data_pl] = discrete_x_test_batch
                if self.conditional:
                    feed_dict[self.real_data_label_pl] = label_data_random[b * self.batch_size: (b + 1) * self.batch_size]

                summary_result, _ = self.sess.run([self.c_vae_summary, self.c_v_op_pre], feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_result, global_id)

                summary_result, _ = self.sess.run([self.d_vae_summary, self.d_v_op_pre], feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_result, global_id)

                real_data, rec_data = self.sess.run([self.c_real_data_pl, self.c_decoded_output], feed_dict=feed_dict)
                c_real_data_lst.append(real_data)
                c_rec_data_lst.append(rec_data)

                real_data, rec_data = self.sess.run([self.d_real_data_pl, self.d_decoded_output], feed_dict=feed_dict)
                d_real_data_lst.append(real_data)
                d_rec_data_lst.append(rec_data)

                global_id += 1

        np.savez('data/rec/vae.npz', c_real=np.vstack(c_real_data_lst), c_rec=np.vstack(c_rec_data_lst),
                                     d_real=np.vstack(d_real_data_lst), d_rec=np.vstack(d_rec_data_lst))

        # saving the pre-trained model
        save_path = os.path.join(self.checkpoint_dir, "pretrain_vae_{}".format(global_id))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save(global_id=global_id - 1, model_name='m3gan', checkpoint_dir=save_path)
        print('finish the pretrain model saving')

        print('finish pretraining')

        # jointly training step
        print('start joint training')

        for e in range(self.num_epochs):
            random_idx = np.random.permutation(data_size)
            continuous_x_random = continuous_x[random_idx]
            discrete_x_random = discrete_x[random_idx]
            if self.conditional:
                label_data_random = label_data[random_idx]

            random_idx_ = np.random.permutation(continuous_x_test.shape[0])
            continuous_x_test_batch = continuous_x_test[random_idx_][:self.batch_size, :, :]
            discrete_x_test_batch = discrete_x_test[random_idx_][:self.batch_size, :, :]

            print("training epoch %d" % e)

            for b in range(num_batches):
                feed_dict = {}
                feed_dict[self.c_real_data_pl] = continuous_x_random[b * self.batch_size: (b + 1) * self.batch_size]
                feed_dict[self.c_gen_input_noise_pl] = self.gen_input_noise(self.batch_size, self.time_steps, noise_dim=self.c_noise_dim)
                feed_dict[self.c_vae_test_data_pl] = continuous_x_test_batch
                feed_dict[self.d_real_data_pl] = discrete_x_random[b * self.batch_size: (b + 1) * self.batch_size]
                feed_dict[self.d_gen_input_noise_pl] = self.gen_input_noise(self.batch_size, self.time_steps, noise_dim=self.d_noise_dim)
                feed_dict[self.d_vae_test_data_pl] = discrete_x_test_batch
                if self.conditional:
                    feed_dict[self.real_data_label_pl] = label_data_random[b * self.batch_size: (b + 1) * self.batch_size]

                for _ in range(self.d_rounds):
                    _, d_summary_result = self.sess.run([self.d_d_op, self.d_discriminator_summary], feed_dict=feed_dict)
                    self.summary_writer.add_summary(d_summary_result, global_id)
                    _, c_summary_result = self.sess.run([self.c_d_op, self.c_discriminator_summary], feed_dict=feed_dict)
                    self.summary_writer.add_summary(c_summary_result, global_id)

                for _ in range(self.g_rounds):
                    _, d_summary_result = self.sess.run([self.d_g_op, self.d_generator_summary], feed_dict=feed_dict)
                    self.summary_writer.add_summary(d_summary_result, global_id)
                    _, c_summary_result = self.sess.run([self.c_g_op, self.c_generator_summary], feed_dict=feed_dict)
                    self.summary_writer.add_summary(c_summary_result, global_id)

                for _ in range(self.v_rounds):
                    _, summary_result = self.sess.run([self.d_v_op, self.d_vae_summary], feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_result, global_id)
                    _, summary_result = self.sess.run([self.c_v_op, self.d_vae_summary], feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_result, global_id)

                global_id += 1

			# generation
            if ((e + 1) % self.epoch_ckpt_freq == 0 or e == self.num_epochs - 1):
                data_gen_path = os.path.join("data/fake/", "epoch{}".format(e))
                if not os.path.exists(data_gen_path):
                    os.makedirs(data_gen_path)
                if self.conditional:
                    d_gen_data, c_gen_data = self.generate_data(num_sample=self.c_data_sample.shape[0],
                                                                labels=self.statics_label)
                else:
                    d_gen_data, c_gen_data = self.generate_data(num_sample=self.c_data_sample.shape[0])
                np.savez(os.path.join(data_gen_path, "gen_data.npz"), c_gen_data=c_gen_data, d_gen_data=d_gen_data)
                print('finish generated data saving in epoch ' + str(e))

    def generate_data(self, num_sample, labels):
        d_gen_data = []
        c_gen_data = []
        round_ = num_sample // self.batch_size

        for i in range(round_):
            d_gen_input_noise = self.gen_input_noise(
                num_sample=self.batch_size, T=self.time_steps, noise_dim=self.d_noise_dim)
            c_gen_input_noise = self.gen_input_noise(
                num_sample=self.batch_size, T=self.time_steps, noise_dim=self.c_noise_dim)

            feed_dict = {}
            feed_dict[self.d_gen_input_noise_pl] = d_gen_input_noise
            feed_dict[self.c_gen_input_noise_pl] = c_gen_input_noise
            if self.conditional:
                feed_dict[self.real_data_label_pl] = \
                    labels[i * self.batch_size: (i + 1) * self.batch_size]

            d_gen_data_, c_gen_data_ = self.sess.run([self.d_gen_decoded, self.c_gen_decoded], feed_dict=feed_dict)
            d_gen_data.append(np.stack(d_gen_data_, axis=1))
            c_gen_data.append(np.stack(c_gen_data_, axis=1))

        d_gen_data = np.concatenate(d_gen_data, axis=0)
        c_gen_data = np.concatenate(c_gen_data, axis=0)

        return d_gen_data, c_gen_data



