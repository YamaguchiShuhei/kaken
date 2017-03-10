import tensorflow as tf
import numpy as np


from nn.model import bi_lstm
from nn.model import f_lstm
from nn.model import lstm_w5
from nn.model import w5_lstm
import batch
from nn import cross_acc
from nn import lattice_cross_acc

class Model:
    def __init__(self, params):
        self.params = params

    def model_build(self, sess):
        with tf.variable_scope('inputs'):
            self.x = tf.placeholder(tf.int32, [None, None], 'text')
            self.y_ = tf.placeholder(tf.int32, [None, None], 'true_label')
            self.l = tf.placeholder(tf.int32, [None], 'sentence_len')
            self.b_ = tf.placeholder(tf.float32, [None, None], 'true_b')
            self.m_ = tf.placeholder(tf.float32, [None, None], 'true_m')
            self.e_ = tf.placeholder(tf.float32, [None, None], 'true_e')
            self.s_ = tf.placeholder(tf.float32, [None, None], 'true_s')
        self.keep_prob_cv = tf.placeholder(tf.float32)

        with tf.variable_scope('model'):
            self.y, self.b, self.m, self.e, self.s =  bi_lstm.bi_lstm(self.params, self.x, self.l, self.keep_prob_cv)
            self.y_one = tf.one_hot(self.y_, 6)
            
            self.loss, self.accuracy = cross_acc.cross_acc(self.params, self.y, self.y_one, self.l)
            self.b_loss, self.b_accuracy = lattice_cross_acc.lattice_cross_acc(self.params, self.b, self.b_, self.l)
            self.m_loss, self.m_accuracy = lattice_cross_acc.lattice_cross_acc(self.params, self.m, self.m_, self.l)
            self.e_loss, self.e_accuracy = lattice_cross_acc.lattice_cross_acc(self.params, self.e, self.e_, self.l)
            self.s_loss, self.s_accuracy = lattice_cross_acc.lattice_cross_acc(self.params, self.s, self.s_, self.l)

            self.lattice_loss = self.b_loss + self.m_loss + self.e_loss + self.s_loss
            self.lattice_accuracy = (self.b_accuracy + self.m_accuracy + self.e_accuracy + self.s_accuracy) / 4.0

        with tf.variable_scope('train'):
            self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

            self.b_train_step = tf.train.AdamOptimizer().minimize(self.b_loss)

            # self.lattice_train_step = tf.train.AdamOptimizer().minimize(self.lattice_loss)
            self.lattice_train_step = tf.train.AdamOptimizer().minimize(self.lattice_loss)
        cross_summary = tf.scalar_summary("cross_entropy", self.loss)
        accuracy_summary = tf.scalar_summary("accuracy", self.accuracy)
        lattice_cross_summary = tf.scalar_summary('lattice_cross_entropy', self.lattice_loss)
        lattice_accuracy_summary = tf.scalar_summary("lattice_accuracy", self.lattice_accuracy)
        b_accuracy_summary = tf.scalar_summary('b_accuracy', self.b_accuracy)
        m_accuracy_summary = tf.scalar_summary('m_accuracy', self.m_accuracy)
        e_accuracy_summary = tf.scalar_summary('e_accuracy', self.e_accuracy)
        s_accuracy_summary = tf.scalar_summary('s_accuracy', self.s_accuracy)
        self.merged = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter('data/train', graph=sess.graph)
        self.test_writer = tf.train.SummaryWriter('data/devel')
        
    def model_initialize(self, sess):
        init = tf.initialize_all_variables()
        sess.run(init)
        
    def text_train(self, data, sess, epoch=40000):
        g = batch.batch_generate(data, self.params['batch_size'])
        for i in range(epoch):
            batch_xs, batch_ys, batch_len, batch_b, batch_m, batch_e, batch_s = next(g)
            feed_dict = {self.x:batch_xs, self.y_:batch_ys, self.l:batch_len, self.b_:batch_b, self.m_:batch_m, self.e_:batch_e, self.s_:batch_s, self.keep_prob_cv:0.8}
            _, accuracy, result = sess.run([self.train_step, self.accuracy, self.merged], feed_dict=feed_dict)
            self.train_writer.add_summary(result, i)
            if i/1000 == int(i/1000):
                print(i, accuracy)
        saver = tf.train.Saver()
        saver.save(sess, "./save/model.ckpt")
        print("Model saved in file: modelckpt")

    def train(self, data, sess, epoch=40000):
        g = batch.batch_generate(data, self.params['batch_size'])
        for i in range(epoch):
            batch_xs, batch_ys, batch_len, batch_b, batch_m, batch_e, batch_s = next(g)
            feed_dict = {self.x:batch_xs, self.y_:batch_ys, self.l:batch_len, self.b_:batch_b, self.m_:batch_m, self.e_:batch_e, self.s_:batch_s, self.keep_prob_cv:0.8}
            _, _, accuracy, lattice_accuracy, result = sess.run([self.train_step, self.lattice_train_step, self.accuracy, self.lattice_accuracy, self.merged], feed_dict=feed_dict)
            self.train_writer.add_summary(result, i)
            if i/1000 == int(i/1000):
                print(i, accuracy, lattice_accuracy)
        saver = tf.train.Saver()
        saver.save(sess, "./save/model.ckpt")
        print("Model saved in file: modelckpt")

    def lattice_train(self, data, sess, epoch=40000):
        g = batch.batch_generate(data, self.params['batch_size'])
        for i in range(epoch):
            batch_xs, batch_ys, batch_len, batch_b, batch_m, batch_e, batch_s = next(g)
            feed_dict = {self.x:batch_xs, self.y_:batch_ys, self.l:batch_len, self.b_:batch_b, self.m_:batch_m, self.e_:batch_e, self.s_:batch_s, self.keep_prob_cv:0.8}
            _, accuracy, result = sess.run([self.lattice_train_step, self.lattice_accuracy, self.merged], feed_dict=feed_dict)
            self.train_writer.add_summary(result, i)
            if i/1000 == int(i/1000):
                print(i, accuracy)
        saver = tf.train.Saver()
        saver.save(sess, "./save/model.ckpt")
        print("Model saved in file: modelckpt")

    def _word_count(self, label_list, testdata):
        b = 0
        for i in range(len(testdata.sentence_len)):
            for k in range(testdata.sentence_len[i]):
                if label_list[i][k] in [0, 3]:
                    b += 1
        return b
    
    def evaluate(self, testdata, sess):
        sum_acc = 0
        sum_lattice_acc = 0
        self.ylist = []
        self.y_list = []
        batch_size = self.params['batch_size']
        g = batch.batch_generate(testdata, batch_size)
        for i in range(int(len(testdata.data)/batch_size)):
            batch_xs, batch_ys, batch_len, batch_b, batch_m, batch_e, batch_s = next(g)
            feed_dict = {self.x:batch_xs, self.y_:batch_ys, self.l:batch_len, self.b_:batch_b, self.m_:batch_m, self.e_:batch_e, self.s_:batch_s, self.keep_prob_cv:1.0}
            sum_acc += sess.run(self.accuracy, feed_dict)
            sum_lattice_acc += sess.run(self.lattice_accuracy, feed_dict)
            self.ylist.extend(sess.run(tf.argmax(self.y, 2), feed_dict))
            self.y_list.extend(sess.run(tf.argmax(self.y_one, 2), feed_dict))
                               
        print('accuracy',sum_acc/int(len(testdata.data)/batch_size))
        print('lattice_accuracy',sum_lattice_acc/int(len(testdata.data)/batch_size))
        self.b_think = self._word_count(self.ylist, testdata)
        self.b_truth = self._word_count(self.y_list, testdata)

        correct = 0
        for i in range(len(self.ylist)):
            for k in range(testdata.sentence_len[i]):
                if self.ylist[i][k] == self.y_list[i][k] == 0:
                    for l in range(testdata.sentence_len[i]-k-1):
                        if self.ylist[i][k+l+1] != self.y_list[i][k+l+1]:
                            break
                        elif self.ylist[i][k+l+1] == self.y_list[i][k+l+1] == 2:
                            correct += 1
                            break
                elif self.ylist[i][k] == self.y_list[i][k] == 3:
                    correct += 1
                    
        tekigou = correct / self.b_think
        saigen = correct / self.b_truth
        f_score = 2 * tekigou * saigen / (tekigou+saigen)
        
        print("think " +str(self.b_think), "truth " +str(self.b_truth), "correct " +str(correct))
        print("tekigou " +str(tekigou), "saigen " +str(saigen))
        print("f " +str(f_score))


    def _demo_read(self, textlist, char_id):
        SENT_len = self.params["SENT_len"]
        text_id = []
        text_id.append(char_id["<sta>"])
        for i in range(len(textlist)):
            if textlist[i] in char_id:
                text_id.append(char_id[textlist[i]])
            else:
                text_id.append(len(char_id))
        sent_len = len(text_id)
        text_id.append(char_id["EOS"])

        self.demo_text_id = np.array([text_id for i in range(10) ], dtype=np.int32)
        self.demo_sentence_len = np.array([ sent_len for i in range(10)], dtype=np.int32)

    def demo(self, text, char_id, sess):
        textlist = list(text)
        self._demo_read(textlist, char_id)
        self.demo_predict = sess.run(tf.argmax(self.y,2), feed_dict={self.x:self.demo_text_id,
                                                                     self.l:self.demo_sentence_len,
                                                                     self.keep_prob_cv:1.0})[0][1:-1]
        print('   /', end='')
        for i in range(len(textlist)):
            if self.demo_predict[i] == 2 or self.demo_predict[i] == 3:
                print(textlist[i],end="")
                print("/", end="")
            else:
                print(textlist[i],end='')
        print()

    def restore(self, path, sess):
        saver = tf.train.Saver()
        saver.restore(sess, path)
        print('model restored')
