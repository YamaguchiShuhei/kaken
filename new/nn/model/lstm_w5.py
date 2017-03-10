import tensorflow as tf
from .. import embed
from .. import divide_predict
from .. import lattice_predict
from .. import lstmw5

def lstm_w5(params, x, l, keep_prob_cv):
    char_vec = embed.embed(params, x, keep_prob_cv)
    lstm_out = lstmw5.lstmw5(params, char_vec, l)

    hidden_size = params['hidden_size']

    predict = divide_predict.divide_predict(params, lstm_out, hidden_size*5)
    b_predict = lattice_predict.lattice_predict(lstm_out, hidden_size*5, 'bw', 'bb')
    m_predict = lattice_predict.lattice_predict(lstm_out, hidden_size*5, 'mw', 'mb')
    e_predict = lattice_predict.lattice_predict(lstm_out, hidden_size*5, 'ew', 'eb')
    s_predict = lattice_predict.lattice_predict(lstm_out, hidden_size*5, 'sw', 'sb')

    return predict, b_predict, m_predict, e_predict, s_predict
