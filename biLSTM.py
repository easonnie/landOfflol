from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from datetime import datetime


class BiLSTM:
    def __init__(self, embedding=None, hidden_state_d=100, max_length=80, learning_rate=0.01, dropout_rate=0.5, vocab_size=400001, embedding_d=300, num_classes=2):
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
        self.reversed_vec_data = tf.reverse_sequence(self.vec_data, seq_dim=1, seq_lengths=self.clean_len)

        with tf.variable_scope('left2right'):
            left2right_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_state_d, state_is_tuple=True)
            self.output, self.state = tf.nn.dynamic_rnn(
                left2right_lstm_cell,
                self.vec_data,
                dtype=tf.float32,
                sequence_length=self.len,
            )

        with tf.variable_scope('right2left'):
            right2left_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_state_d, state_is_tuple=True)
            self.reversed_output, self.reversed_state = tf.nn.dynamic_rnn(
                right2left_lstm_cell,
                self.reversed_vec_data,
                dtype=tf.float32,
                sequence_length=self.len,
            )

        self.last = BiLSTM.last_relevant(self.output, self.len)
        self.reversed_last = BiLSTM.last_relevant(self.reversed_output, self.len)

        self.final_output = tf.concat(1, [self.last, self.reversed_last])

        self.dropout_last = tf.nn.dropout(self.final_output, keep_prob=dropout_rate)

        self.weight = tf.Variable(tf.truncated_normal([hidden_state_d * 2, num_classes], stddev=0.1))
        self.bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        self.prediction = tf.nn.softmax(tf.matmul(self.final_output, self.weight) + self.bias)

        self.cost = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(self.dropout_last, self.weight) + self.bias, self.co_label)
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.init_op = tf.initialize_all_variables()

        self.prediction_a = tf.argmax(self.prediction, dimension=1)
        self.prediction_b = tf.argmax(self.co_label, dimension=1)

        self.score = tf.reduce_sum(tf.cast(tf.equal(self.prediction_a, self.prediction_b), dtype=tf.int32)) / tf.size(self.label)

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
        print('Current Cost', name, 'is :', np.sum(cur_cost))
        print('Current Score', name, 'is :', cur_score)

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

    model = BiLSTM(word_embedding)

    p_train_data_path = os.path.join(path, 'datasets/Diy/sst/p_train_data.txt')
    s_dev_data_path = os.path.join(path, 'datasets/Diy/sst/s_dev_data.txt')
    s_test_data_path = os.path.join(path, 'datasets/Diy/sst/s_test_data.txt')

    train_generator = Batch_generator(filename=p_train_data_path, maxlength=80)
    dev_generator = Batch_generator(filename=s_dev_data_path, maxlength=80)
    test_generator = Batch_generator(filename=s_test_data_path, maxlength=80)

    dev_data, dev_length, dev_label = dev_generator.next_batch(-1)
    t_data, t_length, t_label = test_generator.next_batch(-1)

    print('Start Learning')
    for i in range(10000):
        data, length, label = train_generator.next_batch(256)
        model.train(data, length, label)
        if i % 100 == 0:
            model.predict(dev_data, dev_length, dev_label, name='(dev)')
            model.predict(t_data, t_length, t_label, name='(test)')
            print('Current time is :', str(datetime.now()))
            print('Number of epoch learned:', train_generator.epoch)
