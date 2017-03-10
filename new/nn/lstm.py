import tensorflow as tf

def lstm(params, char_vec, seq_len):
    hidden_size = params["hidden_size"]

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias = 0.0, state_is_tuple = True)
    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, char_vec, sequence_length=seq_len, dtype=tf.float32)
    return outputs
