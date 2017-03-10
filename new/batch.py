import random
import numpy as np

def batch_random(data, batch_size):
    random.seed(0)
    random_list = [ x for x in range(len(data.data)) ]
    i = 0
    while True:
        if len(data.data[i:i+batch_size]) == batch_size:
            yield (data.data[i:i+batch_size],
                   data.label[i:i+batch_size],
                   data.sentence_len[i:i+batch_size],
                   data.b_lattice[i:i+batch_size],
                   data.m_lattice[i:i+batch_size],
                   data.e_lattice[i:i+batch_size],
                   data.s_lattice[i:i+batch_size])
            i += batch_size
        else:
            i = 0
            random.shuffle(random_list)

def batch_padding(batch, max_sentence_len):
    padding_batch = []
    for one_batch in batch:
        padding_batch.append(one_batch + [ one_batch[-1] for x in range(max_sentence_len-len(one_batch)) ])
    return padding_batch
            
def batch_generate(data, batch_size):
    g = batch_random(data, batch_size)
    while True:
        x, y, l, b, m, e, s = next(g)
        max_sentence_len = max(map(len,x))
        yield (np.array(batch_padding(x, max_sentence_len), dtype=np.int32),
               np.array(batch_padding(y, max_sentence_len), dtype=np.int32),
               np.array(l, dtype=np.int32),
               np.array(batch_padding(b, max_sentence_len), dtype=np.float32),
               np.array(batch_padding(m, max_sentence_len), dtype=np.float32),
               np.array(batch_padding(e, max_sentence_len), dtype=np.float32),
               np.array(batch_padding(s, max_sentence_len), dtype=np.float32))
