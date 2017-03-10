import tensorflow as tf

def lstmw5(params, char_vec, seq_len):
    hidden_size = params["hidden_size"]

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias = 0.0, state_is_tuple = True)
    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, char_vec, sequence_length=seq_len, dtype=tf.float32)
    outputs_head = outputs[:,0:1,:]
    outputs_tail = outputs[:,tf.shape(outputs)[1]-1:,:]
    output = tf.concat(1, [outputs_head, outputs_head , outputs, outputs_tail, outputs_tail])
    
    output1 = output[:,:tf.shape(output)[1]-4,:]
    output2 = output[:,1:tf.shape(output)[1]-3,:]
    output3 = output[:,2:tf.shape(output)[1]-2,:]
    output4 = output[:,3:tf.shape(output)[1]-1,:]
    output5 = output[:,4:tf.shape(output)[1],:]
    output_concat = tf.concat(2,[output1,output2,output3,output4,output5]) 
    return output_concat

