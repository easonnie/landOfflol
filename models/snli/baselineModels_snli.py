import numpy as np
import tensorflow as tf
from models.base.seqLSTM import BasicSeqModel


class SnliLoader:
    def __init__(self, lstm_step=80, input_d=300, vocab_size=2196018, embedding=None):
        self.raw_premise = tf.placeholder(shape=[None, lstm_step], dtype=tf.int32, name='premise')
        self.premise_length = tf.placeholder(shape=[None], dtype=tf.int32, name='premise_length')

        self.raw_hypothesis = tf.placeholder(shape=[None, lstm_step], dtype=tf.int32, name='hypothesis')
        self.hypothesis_length = tf.placeholder(shape=[None], dtype=tf.int32, name='hypothesis_length')

        # Those operations take too many memory
        # Use cpu for those operations
        with tf.device("/cpu:0"):
            if embedding is not None:
                self.input_embedding = tf.placeholder(dtype=tf.float32, shape=embedding.shape, name='word_embedding')
                self.embedding = tf.Variable(tf.zeros(embedding.shape, dtype=tf.float32))
            else:
                self.embedding = tf.Variable(tf.random_uniform([vocab_size, input_d], minval=-0.05, maxval=0.05))
            self.load_embedding_op = self.embedding.assign(self.input_embedding)

            self.premise = tf.nn.embedding_lookup(self.embedding, self.raw_premise)
            self.hypothesis = tf.nn.embedding_lookup(self.embedding, self.raw_hypothesis)

        self.label = tf.placeholder(shape=[None], dtype=tf.int32)


class SnliBasicLSTM:
    def __init__(self, lstm_step=80, input_d=300, vocab_size=2196018, hidden_d=100, num_class=3, learning_rate=0.001, softmax_keeprate=0.5, embedding=None,
                 **kwargs):
        self.model_info = self.record_info(LSTM_Step=lstm_step,
                                           Word_Dimension=input_d,
                                           Vocabluary_Size=vocab_size,
                                           LSTM_Hidden_Dimension=hidden_d,
                                           Number_Class=num_class,
                                           SoftMax_Keep_Rate=softmax_keeprate,
                                                        ** kwargs)

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
            b = tf.Variable(tf.constant(0.01, shape=[hidden_d * 2]), name='b')
            self.layer1_tanh_output = tf.nn.tanh(tf.nn.xw_plus_b(self.sentence_embedding_output, W, b))

        with tf.variable_scope('layer2-tanh'):
            W = tf.Variable(tf.truncated_normal([hidden_d * 2, hidden_d * 2], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.01, shape=[hidden_d * 2]), name='b')
            self.layer2_tanh_output = tf.nn.tanh(tf.nn.xw_plus_b(self.layer1_tanh_output, W, b))

        with tf.variable_scope('layer3-tanh'):
            W = tf.Variable(tf.truncated_normal([hidden_d * 2, num_class], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.01, shape=[num_class]), name='b')
            self.layer3_tanh_output = tf.nn.tanh(tf.nn.xw_plus_b(self.layer2_tanh_output, W, b))

        self.softmax_output = tf.nn.softmax(self.layer3_tanh_output)
        self.prediction = tf.argmax(self.softmax_output, dimension=1)

        self.cost = tf.nn.sparse_softmax_cross_entropy_with_logits(tf.nn.dropout(self.layer3_tanh_output, keep_prob=softmax_keeprate), self.input_loader.label)

        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.init_op = tf.initialize_all_variables()
        self.sess = tf.Session()

    def load_embedding(self, embedding=None):
        if embedding is None:
            print('No embedding specified. Use random embedding.')
        else:
            print('Load embedding.', 'Vocabulary size:', embedding.shape[0], 'Word dimension', embedding.shape[1])
            self.sess.run(self.input_loader.load_embedding_op, feed_dict={self.input_loader.input_embedding: embedding})

    def record_info(self, **kwargs):
        return kwargs

    def train(self, feed_dict):
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def predict(self, feed_dict):
        y_pred = feed_dict[self.input_loader.label]
        out_pred, out_cost = self.sess.run((self.prediction, self.cost), feed_dict=feed_dict)
        accuracy = np.sum(y_pred == out_pred) / len(y_pred)
        # print('Current accuracy:', accuracy)
        # print('Current cost:', np.sum(out_cost) / len(out_cost))
        return accuracy, (np.sum(out_cost) / len(out_cost))

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


if __name__ == '__main__':
    model = SnliBasicLSTM(Name='BasicLSTM', Embedding='GLOVE')
    print(str(model.model_info))
