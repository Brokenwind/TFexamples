import tensorflow as tf
import numpy as np

n_steps = 2
n_inputs = 3
n_neurons = 5
n_layers = 3

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
seq_length = tf.placeholder(tf.int32, [None])

layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,
                                      activation=tf.nn.relu)
          for layer in range(n_layers)]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32, sequence_length=seq_length)

init = tf.global_variables_initializer()

X_batch = np.array([
    # step 0     step 1
    [[0, 1, 2], [9, 8, 7]],  # instance 1
    [[3, 4, 5], [0, 0, 0]],  # instance 2 (padded with zero vectors)
    [[6, 7, 8], [6, 5, 4]],  # instance 3
    [[9, 0, 1], [3, 2, 1]],  # instance 4
])

seq_length_batch = np.array([2, 1, 2, 2])

with tf.Session() as sess:
    init.run()
    outputs_val, states_val = sess.run(
        [outputs, states], feed_dict={X: X_batch, seq_length: seq_length_batch})
    print("outputs_val.shape:", outputs, "states_val.shape:", states)
    print("outputs_val:", outputs_val, "states_val:", states_val)