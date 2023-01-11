## discriminative score from timegan: https://github.com/jsyoon0823/TimeGAN/blob/master/metrics/discriminative_metrics.py

# Necessary Packages
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

def discriminative_score_metrics (ori_data, generated_data):
  # Initialization on the Graph
  tf.reset_default_graph()
         
  ## Builde a post-hoc RNN discriminator network 
  X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x")
  X_hat = tf.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x_hat")
    
  T = tf.placeholder(tf.int32, [None], name = "myinput_t")
  T_hat = tf.placeholder(tf.int32, [None], name = "myinput_t_hat")
    
  # discriminator function
  def discriminator (x, t):
    with tf.variable_scope("discriminator", reuse = tf.AUTO_REUSE) as vs:
      d_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'd_cell')
      d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, x, dtype=tf.float32, sequence_length = t)
      y_hat_logit = tf.contrib.layers.fully_connected(d_last_states, 1, activation_fn=None) 
      y_hat = tf.nn.sigmoid(y_hat_logit)
      d_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
    
    return y_hat_logit, y_hat, d_vars
    
  y_logit_real, y_pred_real, d_vars = discriminator(X, T)
  y_logit_fake, y_pred_fake, _ = discriminator(X_hat, T_hat)
        
  # Loss for the discriminator
  d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit_real, labels = tf.ones_like(y_logit_real)))
  d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit_fake, labels = tf.zeros_like(y_logit_fake)))
  d_loss = d_loss_real + d_loss_fake
    
  # optimizer
  d_solver = tf.train.AdamOptimizer().minimize(d_loss, var_list = d_vars)
        
  ## Train the discriminator   
  # Start session and initialize
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
    
  # Training step
  for itt in range(iterations):
          
    # Batch setting
    X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
    X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)
          
    # Train discriminator
    _, step_d_loss = sess.run([d_solver, d_loss], 
                              feed_dict={X: X_mb, T: T_mb, X_hat: X_hat_mb, T_hat: T_hat_mb})            
    
  ## Test the performance on the testing set    
  y_pred_real_curr, y_pred_fake_curr = sess.run([y_pred_real, y_pred_fake], 
                                                feed_dict={X: test_x, T: test_t, X_hat: test_x_hat, T_hat: test_t_hat})
    
  y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis = 0))
  y_label_final = np.concatenate((np.ones([len(y_pred_real_curr),]), np.zeros([len(y_pred_fake_curr),])), axis = 0)
    
  # Compute the accuracy
  discriminative_score = accuracy_score(y_label_final, (y_pred_final>0.5))
    
  return discriminative_score  