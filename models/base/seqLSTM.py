import tensorflow as tf


class BasicSeqModel:
    def __init__(self, input_, length_, hidden_state_d, name, cell=None, input_keep_rate=1.0, output_keep_rate=1.0):
        """
        lstm_step, input_d, hidden_state_d
        :param name:
        :return:
        self.input  (shape=[None, lstm_step, input_d], dtype=tf.float32, name='input')
        self.length (shape=[None], dtype=tf.int32, name='length')
        """
        with tf.variable_scope(name):
            self.input = input_
            self.length = length_

            self.reverse_input = tf.reverse_sequence(self.input, seq_dim=1, seq_lengths=tf.cast(self.length, tf.int64))

            # default : LSTM
            if cell is None:
                cell_f = tf.nn.rnn_cell.BasicLSTMCell(hidden_state_d, state_is_tuple=True)
                cell_r = tf.nn.rnn_cell.BasicLSTMCell(hidden_state_d, state_is_tuple=True)

            if len(cell) > 1:
                cell_f, cell_r = cell
            else:
                cell_f = cell[0]
                cell_r = cell[0]

            with tf.variable_scope('forward'):
                self.output, self.last_state = tf.nn.dynamic_rnn(
                    cell_f,
                    tf.nn.dropout(self.input, input_keep_rate),
                    dtype=tf.float32,
                    sequence_length=self.length,
                )
                self.last = tf.nn.dropout(BasicSeqModel.last_relevant(self.output, self.length),
                                          output_keep_rate)

            with tf.variable_scope('backward'):
                self.reverse_output, self.reverse_last_state = tf.nn.dynamic_rnn(
                    cell_r,
                    tf.nn.dropout(self.reverse_input, input_keep_rate),
                    dtype=tf.float32,
                    sequence_length=self.length,
                )
                self.reverse_last = tf.nn.dropout(BasicSeqModel.last_relevant(self.reverse_output, self.length),
                                                  output_keep_rate)


    @staticmethod
    def last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant


class SeqLoader:
    def __init__(self, lstm_step, input_d, name):
        with tf.name_scope(name):
            self.input = tf.placeholder(shape=[None, lstm_step, input_d], dtype=tf.float32, name='input')
            self.length = tf.placeholder(shape=[None], dtype=tf.int32, name='length')