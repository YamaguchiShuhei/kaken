import tensorflow as tf

import reverse_0tow
import batch
import data
import nn.model.f_lstm
import model
params = {"FREQ_times":10,"SENT_len":300,"embedding_size":100,"batch_size":10,"CHAR_size":4000,"hidden_size":150, 'LABEL_size':6}

if __name__ == '__main__':
    trainpath = './text_corpus/train'
    testpath = './text_corpus/devel'
    webpath = './web_corpus/devel'
    wikipath = './wiki_corpus/10%train'
    traindata = data.Dataset(params)
    traindata.read(trainpath)
    testdata = data.Dataset(params)
    testdata.read(testpath, traindata.char_id)
    webdata = data.Dataset(params)
    webdata.read(webpath, traindata.char_id)
    wikidata = data.Dataset(params)
    wikidata.read(wikipath, traindata.char_id)
    print('data loaded')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    model = model.Model(params)
    model.model_build(sess)
    print('model built')    
    model.model_initialize(sess)
    print('model initialize')

    
    # model.text_train(traindata, sess, 40000)
    # print('train finished')
    # model.evaluate(testdata, sess)
    # print('only text train')
    
    # model.train(traindata, sess, 40000)
    # print('train finished')
    # model.evaluate(testdata, sess)
    # print('lattice and text at the same time')
