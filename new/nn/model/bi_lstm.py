import tensorflow as tf
from .. import embed
from .. import divide_predict
from .. import lattice_predict
from .. import bilstm

def bi_lstm(params, x, l, keep_prob_cv):
    char_vec = embed.embed(params, x, keep_prob_cv)
    lstm_out = bilstm.bilstm(params, char_vec, l)

    hidden_size = params['hidden_size']
    
    predict = divide_predict.divide_predict(params, lstm_out, hidden_size*2)
    b_predict = lattice_predict.lattice_predict(lstm_out, hidden_size*2, 'bw', 'bb')
    m_predict = lattice_predict.lattice_predict(lstm_out, hidden_size*2, 'mw', 'mb')
    e_predict = lattice_predict.lattice_predict(lstm_out, hidden_size*2, 'ew', 'eb')
    s_predict = lattice_predict.lattice_predict(lstm_out, hidden_size*2, 'sw', 'sb')

    return predict, b_predict, m_predict, e_predict, s_predict
