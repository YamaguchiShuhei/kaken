import tensorflow as tf

def divide_predict(params, lstm_out, hidden_size):
    LABEL_size = params['LABEL_size']
    batch_size = params['batch_size']
    lstm_out_reshape = tf.reshape(lstm_out, [-1, hidden_size])
    with tf.variable_scope('divide_weight'):
        W = tf.Variable(tf.random_uniform([hidden_size, LABEL_size], -1.0, 1.0))
    with tf.variable_scope('divide_bias'):
        b = tf.Variable(tf.random_uniform([LABEL_size], -1.0, 1.0))
    with tf.name_scope('divide_predict'):
        y = tf.nn.softmax(tf.matmul(lstm_out_reshape, W) + b)
        yreshape = tf.reshape(y, tf.concat(0,[tf.shape(lstm_out)[:2],[LABEL_size]]))
    return yreshape
