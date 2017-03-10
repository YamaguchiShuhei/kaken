import tensorflow as tf

def lattice_predict(lstm_out, hidden_size, name_w=None, name_b=None):
    lstm_out_reshape = tf.reshape(lstm_out, [-1, hidden_size])  
    with tf.variable_scope(name_w):
        W = tf.Variable(tf.random_uniform([hidden_size,1], -1.0, 1.0))
    with tf.variable_scope(name_b):
        b  = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    y = tf.nn.sigmoid(tf.matmul(lstm_out_reshape, W) + b)
    yreshape = tf.reshape(y, tf.shape(lstm_out)[:2])
    return yreshape
