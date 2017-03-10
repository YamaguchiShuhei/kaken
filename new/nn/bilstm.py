import tensorflow as tf

def bilstm(params, char_vec, seq_len):
    hidden_size = params["hidden_size"]

    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias = 0.0, state_is_tuple = True)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias = 0.0, state_is_tuple = True)
    outputs, _  = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, char_vec, sequence_length=seq_len, dtype=tf.float32)
    output = tf.concat(2, outputs)
    return output
