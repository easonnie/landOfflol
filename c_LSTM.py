from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from datetime import datetime
import time


def fine_grained_score(prediction, label_):
    if len(prediction) != len(label_):
        print('Length not match!')
        return

    num = len(label_)
    hit = 0
    for p, l in zip(prediction, label_):
        if 0 <= l <= 0.2 and 0 <= p <= 0.2:
            hit += 1
        elif 0.2 < l <= 0.4 and 0.2 < p <= 0.4:
            hit += 1
        elif 0.4 < l <= 0.6 and 0.4 < p <= 0.6:
            hit += 1
        elif 0.6 < l <= 0.8 and 0.6 < p <= 0.8:
            hit += 1
        elif 0.8 < l <= 1.0 and 0.8 < p <= 1.0:
            hit += 1

    return hit / num


class CLSTM:
    def __init__(self, embedding=None, hidden_state_d=100, max_length=80, learning_rate=0.001, dropout_rate=0.5,
                 vocab_size=400001, embedding_d=300, num_classes=2, n_gram=3, n_feature=100):
        self.data = tf.placeholder(dtype=tf.int32, shape=[None, max_length])
        self.len = tf.placeholder(dtype=tf.int32, shape=[None])
        self.label = tf.placeholder(dtype=tf.float32, shape=[None])

        self.neg_label = 1 - self.label

        self.co_label = tf.transpose(tf.reshape(tf.concat(0, [self.label, self.neg_label]), [2, -1]))

        self.init_embedding(embedding, vocab_size, embedding_d)

        # filter len to maxlength
        self.maxlen = tf.cast(tf.fill([tf.shape(self.len)[0]], max_length), tf.int64)
        self.filter = tf.less_equal(tf.cast(self.len, tf.int64), self.maxlen)
        self.clean_len = tf.select(self.filter, tf.cast(self.len, tf.int64), self.maxlen)

        self.vec_data = tf.nn.embedding_lookup(self.embedding, self.data)

        # conv filter
        self.conv_filter = tf.Variable(tf.random_uniform([1, n_gram, embedding_d, n_feature], -0.25, 0.25, dtype=tf.float32))
        self.conv_b = tf.Variable(tf.constant(0.1, shape=[n_feature]), name="b")
        self.conv_vec_data = tf.nn.relu(tf.nn.bias_add(tf.reshape(tf.nn.conv2d(tf.reshape(self.vec_data, [-1, 1, max_length, embedding_d]), self.conv_filter, strides=[1, 1, 1, 1], padding='SAME'), [-1, max_length, n_feature]), self.conv_b))

        self.reversed_conv_vec_data = tf.reverse_sequence(self.conv_vec_data, seq_dim=1, seq_lengths=self.clean_len)

        with tf.variable_scope('left2right'):
            left2right_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_state_d, state_is_tuple=True)
            self.output, self.state = tf.nn.dynamic_rnn(
                left2right_lstm_cell,
                self.conv_vec_data,
                dtype=tf.float32,
                sequence_length=self.len,
            )

        with tf.variable_scope('right2left'):
            right2left_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_state_d, state_is_tuple=True)
            self.reversed_output, self.reversed_state = tf.nn.dynamic_rnn(
                right2left_lstm_cell,
                self.reversed_conv_vec_data,
                dtype=tf.float32,
                sequence_length=self.len,
            )

        self.last = CLSTM.last_relevant(self.output, self.len)
        self.reversed_last = CLSTM.last_relevant(self.reversed_output, self.len)

        self.final_output = tf.concat(1, [self.last, self.reversed_last])

        self.dropout_last = tf.nn.dropout(self.final_output, keep_prob=dropout_rate)

        self.weight = tf.Variable(tf.truncated_normal([hidden_state_d * 2, num_classes], stddev=0.1))
        self.bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        self.prediction = tf.nn.softmax(tf.matmul(self.final_output, self.weight) + self.bias)

        self.cost = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(self.dropout_last, self.weight) + self.bias,
                                                            self.co_label)
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.init_op = tf.initialize_all_variables()

        self.prediction_a = tf.argmax(self.prediction, dimension=1)
        self.prediction_b = tf.argmax(self.co_label, dimension=1)

        self.score = tf.reduce_sum(tf.cast(tf.equal(self.prediction_a, self.prediction_b), dtype=tf.int32)) / tf.size(
            self.label)

        self.sess = tf.Session()
        self.sess.run(self.init_op)

    @staticmethod
    def last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant

    def init_embedding(self, embedding=None, vocab_size=400001, embedding_d=300):
        if embedding is None:
            self.embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_d]))
        else:
            self.embedding = tf.Variable(embedding, [vocab_size, embedding_d], dtype=tf.float32)

    def train(self, batch_data, batch_length, batch_label):
        self.sess.run(self.train_op, feed_dict={self.data: batch_data, self.len: batch_length, self.label: batch_label})

    def predict(self, batch_data, batch_length, batch_label, name=None):
        d_prediction, d_label, p_a, p_b, cur_score, cur_cost = self.sess.run((self.prediction, self.co_label, self.prediction_a, self.prediction_b, self.score, self.cost), feed_dict={self.data: batch_data, self.len: batch_length, self.label: batch_label})
        print('Current Binary Cost', name, 'is :', np.sum(cur_cost))
        print('Current Binary Score', name, 'is :', cur_score)
        return cur_score

    def predict_fg(self, batch_data, batch_length, batch_label, name=None):
        d_prediction, cur_cost = self.sess.run((self.prediction, self.cost), feed_dict={self.data: batch_data, self.len: batch_length, self.label: batch_label})
        print('Current FG Cost', name, 'is :', np.sum(cur_cost))
        cur_fg_score = fine_grained_score(d_prediction[:, 0].ravel(), batch_label)
        print('Current FG Score', name, 'is :', cur_fg_score)
        return cur_fg_score

    def close(self):
        self.sess.close()


from dict_builder import voc_builder
from Batch_generator import Batch_generator
import os

if __name__ == '__main__':
    print('Yo Yo!')
    print('Embedding Loading.')
    print('Current time is :', str(datetime.now()))
    print('Loading Glove embedding matrix')

    path = os.getcwd()
    _, word_embedding = voc_builder(os.path.join(path, 'datasets/Glove/glove.6B.300d.txt'))

    print('Finish Loading')
    print('Current time is :', str(datetime.now()))
    print(word_embedding.shape)

    model = CLSTM(word_embedding)

    p_train_data_path = os.path.join(path, 'datasets/Diy/sst/p_train_data.txt')

    s_dev_data_path = os.path.join(path, 'datasets/Diy/sst/s_binary_dev_data.txt')
    s_test_data_path = os.path.join(path, 'datasets/Diy/sst/s_binary_test_data.txt')

    fine_s_dev_data_path = os.path.join(path, 'datasets/Diy/sst/s_dev_data.txt')
    fine_s_test_data_path = os.path.join(path, 'datasets/Diy/sst/s_test_data.txt')

    train_generator = Batch_generator(filename=p_train_data_path, maxlength=80)

    dev_generator = Batch_generator(filename=s_dev_data_path, maxlength=80)
    test_generator = Batch_generator(filename=s_test_data_path, maxlength=80)

    dev_data, dev_length, dev_label = dev_generator.next_batch(-1)
    t_data, t_length, t_label = test_generator.next_batch(-1)

    fine_dev_generator = Batch_generator(filename=fine_s_dev_data_path, maxlength=80)
    fine_test_generator = Batch_generator(filename=fine_s_test_data_path, maxlength=80)

    f_dev_data, f_dev_length, f_dev_label = fine_dev_generator.next_batch(-1)
    f_t_data, f_t_length, f_t_label = fine_test_generator.next_batch(-1)

    # Write to file
    MODEL_NAME = 'c_LSTM-SST'
    out_root = 'results'
    timestamp = '{0:%Y-%m-%d-%H:%M:%S}'.format(datetime.now())

    out_dir = os.path.abspath(os.path.join(os.path.curdir, out_root, '-'.join(['result', MODEL_NAME, timestamp])))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    meta_result_file = os.path.abspath(os.path.join(out_dir, "meta.txt"))

    print("Writing to {}\n".format(out_dir))

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    meta_file = open(meta_result_file, 'w', encoding='utf-8')
    saver = tf.train.Saver(tf.all_variables())

    # important benchmark
    benchmark_b = 0.86
    benchmark_f = 0.45

    print('Start Learning')
    for i in range(10000):
        data, length, label = train_generator.next_batch(64)
        model.train(data, length, label)

        if i % 100 == 0:
            cur_accuracy = model.predict(dev_data, dev_length, dev_label, name='(dev)')
            test_accuracy = model.predict(t_data, t_length, t_label, name='(test)')

            cur_fine_accuracy = model.predict_fg(f_dev_data, f_dev_length, f_dev_label, name='(dev)')
            test_fine_accuracy = model.predict_fg(f_t_data, f_t_length, f_t_label, name='(test)')

            print('Current time is :', str(datetime.now()))
            print('Number of epoch learned:', train_generator.epoch)

            if cur_accuracy > benchmark_b or cur_fine_accuracy > benchmark_f:
                benchmark_b = cur_accuracy
                benchmark_f = cur_fine_accuracy

                tstp = str(int(time.time())) + '.ckpt'
                path = saver.save(model.sess, os.path.join(checkpoint_dir, tstp))
                meta_file.write('ckp:' + path + '\n')
                meta_file.write('Binary - dev accuracy: ' + str(cur_accuracy) + ' test accuracy: ' + str(test_accuracy) + '\n')
                meta_file.write('Fine - dev accuracy: ' + str(cur_fine_accuracy) + ' test accuracy: ' + str(test_fine_accuracy) + '\n')
                meta_file.write(str(datetime.now()) + '\n')
                print("Saved model checkpoint to {} with accuracy {}\n".format(path, cur_accuracy))