import tensorflow as tf

def w5lstm(params, char_vec, seq_len):
    hidden_size = params["hidden_size"]
    embedding_size = params['embedding_size']

    char_vec_head = char_vec[:,0:1,:]
    char_vec_tail = char_vec[:,tf.shape(char_vec)[1]-1:,:]
    char_vec_mod = tf.concat(1, [char_vec_head, char_vec_head , char_vec, char_vec_tail, char_vec_tail])

    char_vec_1 = tf.concat(1, [char_vec_head, char_vec_head , char_vec[:,:tf.shape(char_vec)[1]-2,:]])
    char_vec_2 = tf.concat(1, [char_vec_head, char_vec[:,:tf.shape(char_vec)[1]-1,:]])
    char_vec_3 = char_vec
    char_vec_4 = tf.concat(1, [char_vec[:,1:,:], char_vec_tail])
    char_vec_5 = tf.concat(1, [char_vec[:,2:,:], char_vec_tail, char_vec_tail])
    char_vec_concat = tf.concat(2, [char_vec_1, char_vec_2, char_vec_3, char_vec_4, char_vec_5])
    
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias = 0.0, state_is_tuple = True)
    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, char_vec_concat, sequence_length=seq_len, dtype=tf.float32)

    return outputs

