from models.snli.baselineModels_snli import SnliBasicLSTM, SnliLoader
from models.base.seqLSTM import BasicSeqModel
import models.base.util as baseU
import tensorflow as tf


class biLSTM(SnliBasicLSTM):
    def __init__(self, lstm_step=80, input_d=300, vocab_size=2196018, hidden_d=100, num_class=3, learning_rate=0.001,
                 softmax_keeprate=0.75, lstm_input_keep_rate=1.0, lstm_output_keep_rate=0.9, embedding=None, **kwargs):
        self.model_info = baseU.record_info(LSTM_Step=lstm_step,
                                            Word_Dimension=input_d,
                                            Vocabluary_Size=vocab_size,
                                            LSTM_Hidden_Dimension=hidden_d,
                                            Number_Class=num_class,
                                            SoftMax_Keep_Rate=softmax_keeprate,
                                            LSTM_Input_Keep_Rate=lstm_input_keep_rate,
                                            LSTM_Output_Keep_Rate=lstm_output_keep_rate,
                                            kwargs=kwargs)

        self.input_loader = SnliLoader(lstm_step, input_d, vocab_size, embedding)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_d, state_is_tuple=True)
        lstm_cell_r = tf.nn.rnn_cell.BasicLSTMCell(hidden_d, state_is_tuple=True)

        basic_seq_lstm_premise = BasicSeqModel(input_=self.input_loader.premise,
                                               length_=self.input_loader.premise_length,
                                               hidden_state_d=hidden_d,
                                               name='premise-lstm', cell=[lstm_cell, lstm_cell_r],
                                               input_keep_rate=lstm_input_keep_rate,
                                               output_keep_rate=lstm_output_keep_rate)

        basic_seq_lstm_hypothesis = BasicSeqModel(input_=self.input_loader.hypothesis,
                                                  length_=self.input_loader.hypothesis_length,
                                                  hidden_state_d=hidden_d,
                                                  name='hypothesis-lstm', cell=[lstm_cell, lstm_cell_r],
                                                  input_keep_rate=lstm_input_keep_rate,
                                                  output_keep_rate=lstm_output_keep_rate)

        self.premise_lstm_last = basic_seq_lstm_premise.last
        self.r_premise_lstm_last = basic_seq_lstm_premise.reverse_last

        self.hypothesis_lstm_last = basic_seq_lstm_hypothesis.last
        self.r_hypothesis_lstm_last = basic_seq_lstm_hypothesis.reverse_last

        self.premise_embedding = tf.concat(1, [self.premise_lstm_last, self.r_premise_lstm_last])
        self.hypothesis_embedding = tf.concat(1, [self.hypothesis_lstm_last, self.r_hypothesis_lstm_last])

        self.sentence_embedding_output = tf.concat(1, [self.premise_embedding, self.hypothesis_embedding,
                                                       tf.abs(self.premise_embedding - self.hypothesis_embedding),
                                                       tf.mul(self.premise_embedding, self.hypothesis_embedding)])

        with tf.variable_scope('layer1'):
            W = tf.Variable(tf.random_uniform([hidden_d * 8, hidden_d * 8], minval=-0.02, maxval=0.02), name='W')
            b = tf.Variable(tf.constant(0.02, shape=[hidden_d * 8]), name='b')
            self.layer1_output = tf.nn.tanh(tf.nn.xw_plus_b(self.sentence_embedding_output, W, b))

        with tf.variable_scope('layer2'):
            W = tf.Variable(tf.random_uniform([hidden_d * 8, hidden_d * 8], minval=-0.02, maxval=0.02), name='W')
            b = tf.Variable(tf.constant(0.02, shape=[hidden_d * 8]), name='b')
            self.layer2_output = tf.nn.tanh(tf.nn.xw_plus_b(self.layer1_output, W, b))

        with tf.variable_scope('layer3'):
            W = tf.Variable(tf.random_uniform([hidden_d * 8, num_class], minval=-0.02, maxval=0.02), name='W')
            b = tf.Variable(tf.constant(0.02, shape=[num_class]), name='b')
            self.layer3_output = tf.nn.tanh(tf.nn.xw_plus_b(self.layer2_output, W, b))

        self.softmax_output = tf.nn.softmax(self.layer3_output)
        self.prediction = tf.argmax(self.softmax_output, dimension=1)

        self.cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf.nn.dropout(self.layer3_output, keep_prob=softmax_keeprate), self.input_loader.label)

        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.init_op = tf.initialize_all_variables()
        self.sess = tf.Session()