import tensorflow as tf


class BasicSeqLSTM:
    def __init__(self, input_, length_, hidden_state_d, name):
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

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_state_d, state_is_tuple=True)
            self.output, self.last_state = tf.nn.dynamic_rnn(
                lstm_cell,
                self.input,
                dtype=tf.float32,
                sequence_length=self.length,
            )
            self.last = BasicSeqLSTM.last_relevant(self.output, self.length)

    @staticmethod
    def last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant