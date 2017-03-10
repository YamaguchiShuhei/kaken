import tensorflow as tf

import reverse_0tow
import batch
import data
import nn.model.f_lstm
import model
params = {"FREQ_times":10,"SENT_len":300,"embedding_size":100,"batch_size":10,"CHAR_size":6000,"hidden_size":150, 'LABEL_size':6}

if __name__ == '__main__':
    trainpath = 'text train no path'
    testpath = 'text test no path'
    webpath = 'web no path'
    wikipath = 'wiki no path'
    print('wiki load')
    wikidata = data.Dataset(params)
    wikidata.read(wikipath)
    print('wiki loaded')
    traindata = data.Dataset(params)
    traindata.read(trainpath, wikidata.char_id)
    testdata = data.Dataset(params)
    testdata.read(testpath, wikidata.char_id)
    webdata = data.Dataset(params)
    webdata.read(webpath, wikidata.char_id)
    print('data loaded')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)  
    model = model.Model(params)
    model.model_build(sess)
    print('model built')
    model.model_initialize(sess)
    print('model initialize')


    # model.lattice_train(wikidata, sess, 300000)
    # print('wiki_lattice finished')
    # model.train(traindata, sess, 40000)
    # print('all_train finished')
    # model.evaluate(testdata, sess)
    
    # model.train(traindata, sess, 40000)
    # print('train finished')
    # model.evaluate(testdata, sess)
    # print('lattice and text at the same time')
    model.restore('./calc_data/lattice2textlattice2divide_data/all_finished_save/model.ckpt', sess)
    wiki = [ x for x in open('hoge', 'r') ]
    for text in wiki:
        model.demo(text.strip(), wikidata.char_id, sess)
        
