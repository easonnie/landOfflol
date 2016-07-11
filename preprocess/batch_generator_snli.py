from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class BatchGenerator:
    def __init__(self, filename, maxlength=None):
        self.filename = filename
        self.file = None
        self.maxlength = maxlength
        self.isOpen = False
        self.epoch = 0
        self.total_size = 0

    def next_batch(self, batch_size, maxlength=None):
        """
        truncate the length to maxlength
        :param batch_size: indicate the size of the batch
        :param maxlength: indicate the max length of the batch size
        """

        if maxlength is None:
            maxlength = self.maxlength

        if not self.isOpen:
            self.file = open(self.filename, 'r')
            self.file.readline()
            self.isOpen = True

        batch_sentence1 = np.empty((0, maxlength), dtype=np.int32)
        batch_length1 = np.empty(0, dtype=np.int32)
        batch_sentence2 = np.empty((0, maxlength), dtype=np.int32)
        batch_length2 = np.empty(0, dtype=np.int32)
        batch_label = np.empty(0, dtype=np.int32)

        # if batch_size == -1, then read the whole file
        while batch_size > 0 or batch_size == -1:
            line = self.file.readline()
            if line == '':
                if batch_size == -1:
                    break
                else:
                    self.epoch += 1
                    self.file.seek(0, 0)
                    self.file.readline() # skip the first line
                    continue

            if batch_size != -1:
                batch_size -= 1

            self.total_size += 1

            line = line.split('|')
            # sentence1 : premise
            s_ids_1 = line[0]

            s_ids_1 = np.fromstring(s_ids_1, dtype=np.int32, sep=' ')
            pad = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0], dtype=np.int32)))
            s_ids_1 = pad(s_ids_1, maxlength)

            batch_sentence1 = np.append(batch_sentence1, [s_ids_1], axis=0)
            s_len_1 = line[1]
            batch_length1 = np.append(batch_length1, np.int32(s_len_1))

            # sentence2 : hypothesis
            s_ids_2 = line[2]

            s_ids_2 = np.fromstring(s_ids_2, dtype=np.int32, sep=' ')
            pad = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0], dtype=np.int32)))
            s_ids_2 = pad(s_ids_2, maxlength)

            batch_sentence2 = np.append(batch_sentence2, [s_ids_2], axis=0)
            s_len_2 = line[3]
            batch_length2 = np.append(batch_length2, np.int32(s_len_2))

            # label
            p_label = line[4]
            batch_label = np.append(batch_label, np.int32(p_label))

        return batch_sentence1, batch_length1, batch_sentence2, batch_length2, batch_label

    def close(self):
        self.epoch = 0
        self.total_size = 0
        self.isOpen = False
        self.file.close()


if __name__ == '__main__':
    generator = BatchGenerator('/Users/Eason/RA/landOfflol/datasets/Diy/snli/test_data.txt', maxlength=10)
    # seems pass the test!
    # print(generator.next_batch(3))