import tensorflow as tf

def embed(params, x, keep_prob_cv):
    embedding_size = params["embedding_size"]
    CHAR_size = params['CHAR_size']
 
    Cembed = tf.Variable(tf.random_uniform([CHAR_size, embedding_size], -1.0, 1.0)) 
    char_vec = tf.nn.embedding_lookup(Cembed, x)
    char_vec_drop = tf.nn.dropout(char_vec, keep_prob_cv)

    return char_vec_drop
