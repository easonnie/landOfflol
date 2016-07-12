import numpy as np
import tensorflow as tf

from models.base.seqLSTM import BasicSeqModel


class SnliLoader:
    def __init__(self, lstm_step=50, input_d=300, vocab_size=400001, embedding=None):
        self.raw_premise = tf.placeholder(shape=[None, lstm_step], dtype=tf.int32, name='premise')
        self.premise_length = tf.placeholder(shape=[None], dtype=tf.int32, name='premise_length')

        self.raw_hypothesis = tf.placeholder(shape=[None, lstm_step], dtype=tf.int32, name='hypothesis')
        self.hypothesis_length = tf.placeholder(shape=[None], dtype=tf.int32, name='hypothesis_length')

        if embedding is None:
            embedding = tf.Variable(tf.random_uniform([vocab_size, input_d], minval=-0.05, maxval=0.05))
        else:
            embedding = tf.Variable(embedding, [vocab_size, input_d], dtype=tf.float32)

        self.premise = tf.nn.embedding_lookup(embedding, self.raw_premise)
        self.hypothesis = tf.nn.embedding_lookup(embedding, self.raw_hypothesis)

        self.label = tf.placeholder(shape=[None], dtype=tf.int32)


class SnliBasicLSTM:
    def __init__(self, lstm_step=50, input_d=300, vocab_size=400001, hidden_d=10, num_class=3, embedding=None):
        self.input_loader = SnliLoader(lstm_step, input_d, vocab_size, embedding)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_d, state_is_tuple=True)
        basic_seq_lstm_premise = BasicSeqModel(input_=self.input_loader.premise,
                                               length_=self.input_loader.premise_length, hidden_state_d=hidden_d,
                                               name='premise-lstm', cell=lstm_cell)
        basic_seq_lstm_hypothesis = BasicSeqModel(input_=self.input_loader.hypothesis,
                                                  length_=self.input_loader.hypothesis_length, hidden_state_d=hidden_d,
                                                  name='hypothesis-lstm', cell=lstm_cell)

        self.premise_lstm_last = basic_seq_lstm_premise.last
        self.hypothesis_lstm_last = basic_seq_lstm_hypothesis.last

        self.sentence_embedding_output = tf.concat(1, [self.premise_lstm_last, self.hypothesis_lstm_last])

        with tf.variable_scope('layer1-tanh'):
            W = tf.Variable(tf.truncated_normal([hidden_d * 2, hidden_d * 2], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[hidden_d * 2]), name='b')
            self.layer1_tanh_output = tf.nn.tanh(tf.nn.xw_plus_b(self.sentence_embedding_output, W, b))

        with tf.variable_scope('layer2-tanh'):
            W = tf.Variable(tf.truncated_normal([hidden_d * 2, hidden_d * 2], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[hidden_d * 2]), name='b')
            self.layer2_tanh_output = tf.nn.tanh(tf.nn.xw_plus_b(self.layer1_tanh_output, W, b))

        with tf.variable_scope('layer3-tanh'):
            W = tf.Variable(tf.truncated_normal([hidden_d * 2, num_class], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_class]), name='b')
            self.layer3_tanh_output = tf.nn.tanh(tf.nn.xw_plus_b(self.layer2_tanh_output, W, b))

        self.softmax_output = tf.nn.softmax(self.layer3_tanh_output)
        self.prediction = tf.argmax(self.softmax_output, dimension=1)

        self.cost = tf.nn.sparse_softmax_cross_entropy_with_logits(self.layer3_tanh_output, self.input_loader.label)

        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)
        self.init_op = tf.initialize_all_variables()
        self.sess = tf.Session()

    def train(self, feed_dict):
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def predict(self, feed_dict):
        y_pred = feed_dict[self.input_loader.label]
        out_pred, out_cost = self.sess.run((self.prediction, self.cost), feed_dict=feed_dict)
        accuracy = np.sum(y_pred == out_pred) / len(y_pred)
        print('Current accuracy:', accuracy)
        print('Current cost:', np.sum(out_cost) / len(out_cost))

    def setup(self):
        self.sess.run(self.init_op)

    def close(self):
        self.sess.close()

    def test(self, feed_dict=None):
        self.setup()
        # d_prem, d_hyp = self.sess.run((self.input_loader.premise, self.input_loader.hypothesis), feed_dict=feed_dict)
        # print(d_prem.shape, d_hyp.shape)
        # d_p_last, d_h_last, d_embd = self.sess.run((self.premise_lstm_last, self.hypothesis_lstm_last, self.sentence_embedding_output), feed_dict=feed_dict)
        # print(d_p_last, d_h_last, d_embd)
        d_pred = self.sess.run(self.prediction, feed_dict=feed_dict)
        print(d_pred)
        # self.close()


if __name__ == '__main__':
    from preprocess.snli.batch_generator_snli import BatchGenerator

    max_length = 50
    basicLSTM = SnliBasicLSTM(lstm_step=max_length)
    dev_batch_generator = BatchGenerator('datasets/Diy/snli/dev_data.txt', maxlength=max_length)
    train_batch_generator = BatchGenerator('datasets/Diy/snli/train_data.txt', maxlength=max_length)

    dev_premise, dev_premise_len, dev_hypothesis, dev_hypothesis_len, dev_label = dev_batch_generator.next_batch(-1)

    in_dev_premise = basicLSTM.input_loader.raw_premise
    in_dev_premise_len = basicLSTM.input_loader.premise_length
    in_dev_hypothesis = basicLSTM.input_loader.raw_hypothesis
    in_dev_hypothesis_len = basicLSTM.input_loader.hypothesis_length
    in_dev_label = basicLSTM.input_loader.label

    in_dev_feed_dit = {in_dev_premise: dev_premise,
                       in_dev_premise_len: dev_premise_len,
                       in_dev_hypothesis: dev_hypothesis,
                       in_dev_hypothesis_len: dev_hypothesis_len,
                       in_dev_label: dev_label}

    i = 0
    basicLSTM.setup()

    while True:
        premise, premise_len, hypothesis, hypothesis_len, label = train_batch_generator.next_batch(128)

        in_premise = basicLSTM.input_loader.raw_premise
        in_premise_len = basicLSTM.input_loader.premise_length
        in_hypothesis = basicLSTM.input_loader.raw_hypothesis
        in_hypothesis_len = basicLSTM.input_loader.hypothesis_length
        in_label = basicLSTM.input_loader.label

        in_feed_dit = {in_premise: premise,
                       in_premise_len: premise_len,
                       in_hypothesis: hypothesis,
                       in_hypothesis_len: hypothesis_len,
                       in_label: label}

        basicLSTM.train(feed_dict=in_feed_dit)
        i += 1

        if i % 100 == 0:
            print('dev:')
            basicLSTM.predict(feed_dict=in_dev_feed_dit)
            print('train:')
            basicLSTM.predict(feed_dict=in_feed_dit)