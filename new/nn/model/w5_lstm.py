import tensorflow as tf
from .. import embed
from .. import divide_predict
from .. import lattice_predict
from .. import w5lstm

def w5_lstm(params, x, l, keep_prob_cv):
    char_vec = embed.embed(params, x, keep_prob_cv)
    lstm_out = w5lstm.w5lstm(params, char_vec, l)

    hidden_size = params['hidden_size']

    predict = divide_predict.divide_predict(params, lstm_out, hidden_size)
    b_predict = lattice_predict.lattice_predict(lstm_out, hidden_size, 'bw', 'bb')
    m_predict = lattice_predict.lattice_predict(lstm_out, hidden_size, 'mw', 'mb')
    e_predict = lattice_predict.lattice_predict(lstm_out, hidden_size, 'ew', 'eb')
    s_predict = lattice_predict.lattice_predict(lstm_out, hidden_size, 'sw', 'sb')

    return predict, b_predict, m_predict, e_predict, s_predict
