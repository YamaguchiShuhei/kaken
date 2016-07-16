import tensorflow as tf
import numpy as np
import collections
import random
import time
import sys
import os

def _read_words(filename):
    f = open(filename, "r")
    line = f.readline()
    list = []
    while line:
        list.append(line.split())
        line = f.readline()
    word_list = []
    for line in list:
        if len(line) > 6:
            word_list.append(line[0])
        elif len(line) == 1:
            word_list.append(line[0])
    return word_list

def _character_id(cha_list):
    counter = collections.Counter(cha_list)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    cha_id = dict(zip(words, range(len(words))))
    return cha_id

def _sentence_list(raw_data, cha_id, SENT_len):
    sentence_list = []
    sentence_label_list = []
    sentence = []
    label = []
    sent_len =[]
    for line in raw_data:
        if line == "EOS":
            sent_len.append(len(sentence))
            if len(sentence) >= SENT_len:
                print("Error!! over SENT")
                exit()
            for k in range(SENT_len):
                if len(sentence) < SENT_len:
                    sentence.append(cha_id[line])
                    label.append([0,0,1])
            sentence_list.append(sentence)
            sentence_label_list.append(label)
            sentence = []
            label = []
        else:
            for i in range(len(list(line))):
                if i == 0:
                    if list(line)[i] in cha_id:
                        sentence.append(cha_id[list(line)[i]])
                    else:
                        sentence.append(len(cha_id)+1)
                    label.append([1,0,0])
                else:
                    if list(line)[i] in cha_id:
                        sentence.append(cha_id[list(line)[i]])
                    else:
                        sentence.append(len(cha_id)+1)
                    label.append([0,1,0])
    return sentence_list, sentence_label_list, sent_len
        

def make_data(data_path, SENT_len):
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "950104")
    train_raw = _read_words(train_path)
    test_raw = _read_words(test_path)

    cha_list = []
    for line in train_raw:
        if line == "EOS":
            cha_list.append(line)
        else:
            cha_list.extend(line)

    cha_id = _character_id(cha_list)
    train_data, train_label, train_sent_len = _sentence_list(train_raw, cha_id, SENT_len)
    test_data, test_label, test_sent_len = _sentence_list(test_raw, cha_id, SENT_len)
    return train_data, train_label, train_sent_len, test_data, test_label, test_sent_len, cha_id, cha_list[train_sent_len[0]+train_sent_len[1]+train_sent_len[2]+3:train_sent_len[0]+train_sent_len[1]+train_sent_len[2]+train_sent_len[3]+3]

def batch_random(data_list, label_list, sent_len, batch_size = 1):
    random_list = []
    for i in range(len(data_list)):
        random_list.append(i)
    random.shuffle(random_list)
    return (np.array([data_list[ random_list[x] ] for x in range(batch_size)], dtype=np.int32),
            np.array([label_list[ random_list[x] ] for x in range(batch_size)], dtype=np.int32),
            np.array([sent_len[ random_list[x] ] for x in range(batch_size)], dtype=np.int32))


if __name__ == "__main__":
    random.seed(0)
    tf.set_random_seed(0)

    train_time = 40000
    LABEL_size = 3
    SENT_len = 300
    CHA_size = 4000
    batch_size = 10
    embed_size = 100
    hidden_size = 150
    triangle = []
    for i in range(SENT_len+1):
        triangle.append([1 for x in range(i)] + [0 for x in range(SENT_len-i)])
    l_look = tf.constant(np.array(triangle, dtype=np.float32))
    
    start_time = time.time()
    print("----read start----")
    data_path = "/home/yamaguchi.13093/syn" 
    train_data, train_label, train_sent_len, test_data, test_label, test_sent_len, cha_id, check = make_data(data_path, SENT_len)

    x = tf.placeholder(tf.int32, [batch_size, SENT_len])
    Wembed = tf.Variable(tf.random_uniform([CHA_size, embed_size], -1.0, 1.0))
    word_vec = tf.nn.embedding_lookup(Wembed, x)
    #word_vec_reshape = tf.reshape(word_vec, [batch_size, SENT_len, embed_size])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias = 0.0)
    inputs = [word_vec[:,time_step,:] for time_step in range(SENT_len)]
    l = tf.placeholder(tf.int32, [batch_size])
    outputs, final_state = tf.nn.rnn(lstm_cell, inputs, dtype=tf.float32, sequence_length=l)
    output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])

    W = tf.Variable(tf.random_uniform([hidden_size, LABEL_size], -1.0, 1.0))
    b = tf.Variable(tf.random_uniform([LABEL_size], -1.0, 1.0))
    y = tf.nn.softmax(tf.matmul(output, W) + b)
    yreshape = tf.reshape(y, [batch_size, SENT_len, LABEL_size])
    y_ = tf.placeholder(tf.float32, [batch_size, SENT_len, LABEL_size])

    W_hist = tf.histogram_summary("weights", W)
    b_hist = tf.histogram_summary("biases", b)

    cross_entropy = tf.reduce_sum(-tf.reduce_sum(y_*tf.log(yreshape), reduction_indices=[2]) * tf.nn.embedding_lookup(l_look, l) / tf.cast(tf.reduce_sum(l), dtype=tf.float32))
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.01, global_step, 2000, 0.9, staircase=True)
    #loss = cross_entropy + 0.00001 * tf.nn.l2_loss(W)
    loss = cross_entropy
    train_step = tf.train.MomentumOptimizer(learning_rate, 0.95).minimize(loss, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(yreshape,2), tf.argmax(y_,2))
    accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float") * tf.nn.embedding_lookup(l_look,l)) / tf.cast(tf.reduce_sum(l), "float")

    cross_summary = tf.scalar_summary("cross_entropy", cross_entropy)    
    accuracy_summary = tf.scalar_summary("accuracy", accuracy)
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter('data/train', graph=sess.graph)

    print("----start train----")
    for i in range(train_time):
        batch_xs, batch_ys, batch_len = batch_random(train_data, train_label, train_sent_len, batch_size)
        sess.run(train_step, feed_dict={x: batch_xs,
                                        y_: batch_ys,
                                        l:batch_len})
        print(i, end=" ")
        print(sess.run(cross_entropy, feed_dict={x:train_data[0:batch_size],
                                                 y_:train_label[0:batch_size],
                                                 l:train_sent_len[0:batch_size]}), end=" ")
        sys.stdout.flush()
        if i/10 == int(i/10):
            print()
            print(i, end=" ")
            foo = sess.run(tf.argmax(yreshape,2), feed_dict={x:train_data[0:batch_size],
                                                             y_:train_label[0:batch_size],
                                                             l:train_sent_len[0:batch_size]})[3]
            for k in range(train_sent_len[3]):
                if foo[k] == 0:
                    print(end=" ")
                    print(check[k], end="")
                else:
                    print(check[k], end="")
            print()

        result = sess.run(merged, feed_dict={x:train_data[0:batch_size],
                                             y_:train_label[0:batch_size],
                                             l:train_sent_len[0:batch_size]})
        train_writer.add_summary(result, i)

    print("----train finish----")
    print("accuracy:",end=" ")
    sum_acc = 0
    ylist = []
    y_list = []
    for i in range(0,len(test_data),batch_size):
        if len(test_data[i:i+batch_size]) == batch_size:
            sum_acc += sess.run(accuracy, feed_dict={x:test_data[i:i+batch_size], y_:test_label[i:i+batch_size], l:test_sent_len[i:i+batch_size]})
    print(sum_acc/int(len(test_data)/batch_size))
            
    end_time = time.time()
    print("calc time: " + str(end_time - start_time))

