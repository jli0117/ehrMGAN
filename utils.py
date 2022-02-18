import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def ones_target(size, min, max):
    return tf.random_uniform([size, 1], min, max, dtype=tf.float32)

def zeros_target(size, min, max):
    return tf.random_uniform([size, 1], min, max, dtype=tf.float32)

def target_flip(size, min, max, ratio):
    ori = np.random.uniform(low=min, high=max, size=[size, 1])
    index = np.random.choice(size, int(size * ratio), replace=False)
    ori[index] = 1 - ori[index]
    return tf.convert_to_tensor(ori, dtype=tf.float32)
