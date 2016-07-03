import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn


class BasicLSTM():
    def __init__(self, embedding=None, hidden_state_d=20, max_length=50, learning_rate=0.01, dropout_rate=1.0, vocab_size=400001, embedding_d=300, num_classes=2):
        self.data = tf.placeholder(dtype=tf.int32, shape=[None, max_length])
        self.len = tf.placeholder(dtype=tf.int32, shape=[None])
        self.label = tf.placeholder(dtype=tf.float32, shape=[None])

        self.neg_label = 1 - self.label

        self.co_label = tf.transpose(tf.reshape(tf.concat(0, [self.label, self.neg_label]), [2, -1]))

        self.init_embedding(embedding, vocab_size, embedding_d)

        self.vec_data = tf.nn.embedding_lookup(self.embedding, self.data)

        lstm_cell = rnn_cell.BasicLSTMCell(hidden_state_d)

        self.output, self.state = rnn.dynamic_rnn(
            lstm_cell,
            self.vec_data,
            dtype=tf.float32,
            sequence_length=self.len,
        )

        self.last = BasicLSTM.last_relevant(self.output, self.len)
        self.dropout_last = tf.nn.dropout(self.last, keep_prob=dropout_rate)

        self.weight = tf.Variable(tf.truncated_normal([hidden_state_d, num_classes], stddev=0.1))
        self.bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        self.prediction = tf.nn.softmax(tf.matmul(self.last, self.weight) + self.bias)

        self.cost = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(self.dropout_last, self.weight) + self.bias, self.co_label)
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.init_op = tf.initialize_all_variables()

        self.prediction_a = tf.argmax(self.prediction, dimension=1)
        self.prediction_b = tf.argmax(self.co_label, dimension=1)

        self.score = tf.reduce_sum(tf.cast(tf.equal(self.prediction_a, self.prediction_b), dtype=tf.int32)) / tf.size(self.label)

        self.sess = tf.Session()
        self.sess.run(self.init_op)

    def last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant

    def init_embedding(self, embedding=None, vocab_size=400001, embedding_d=300):
        if embedding == None:
            self.embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_d]))
        else:
            self.embedding = tf.Variable(embedding, [vocab_size, embedding_d], dtype=tf.float32)

    def train(self, batch_data, batch_length, batch_label):
        self.sess.run(self.train_op, feed_dict={self.data: batch_data, self.len: batch_length, self.label: batch_label})

    def predict(self, batch_data, batch_length, batch_label):
        d_prediction, d_label, p_a, p_b, cur_score, cur_cost = self.sess.run((self.prediction, self.co_label, self.prediction_a, self.prediction_b, self.score, self.cost), feed_dict={self.data: batch_data, self.len: batch_length, self.label: batch_label})
        print('Current Cost is :', np.sum(cur_cost))
        print('Current Score is :', cur_score)

    def close(self):
        self.sess.close()

from dict_builder import voc_builder
from Batch_generator import Batch_generator

if __name__ == '__main__':
    print('Yo Yo!')

    print('Loading Glove embedding matrix')

    path = '/Users/Eason/RA/landOfflol/'
    _, word_embedding = voc_builder(path + 'datasets/Glove/glove.6B.300d.txt')
    print(word_embedding.shape)
    model = BasicLSTM(word_embedding)

    train_generator = Batch_generator(filename='/Users/Eason/RA/landOfflol/datasets/Diy/sst/p_train_data.txt', maxlength=50)
    dev_generator = Batch_generator(filename='/Users/Eason/RA/landOfflol/datasets/Diy/sst/s_dev_data.txt', maxlength=50)

    dev_data, dev_length, dev_label = dev_generator.next_batch(-1)

    for i in range(1000):
        data, length, label = train_generator.next_batch(256)
        model.train(data, length, label)
        if i % 100 == 0:
            model.predict(dev_data, dev_length, dev_label)
