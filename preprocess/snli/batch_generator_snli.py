from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import progressbar as pgb


class BatchGenerator:
    @staticmethod
    def block(file, block_size=65536):
        while True:
            nb = file.read(block_size)
            if not nb:
                break
            yield nb

    @staticmethod
    def whole(file, block_size=65536):
        data = ''
        while True:
            nb = file.read(block_size)
            if not nb:
                break
            data += nb
        return data

    def __init__(self, filename, maxlength=None):
        self.filename = filename
        self.file = None
        self.maxlength = maxlength
        self.isOpen = False
        self.epoch = 0
        self.total_batch = 0
        self.total_num = 0
        self.cur_pointer = 0
        with open(filename, 'r') as f:
            self.total_num = sum(line.count('\n') for line in BatchGenerator.block(f)) - 1
        print('Found data file:', filename, 'Total number:', self.total_num)

    def next_batch(self, batch_size, maxlength=None, save=True):
        """
        truncate the length to maxlength
        :param batch_size: indicate the size of the batch
        :param maxlength: indicate the max length of the batch size
        :param save: whether cached data for dev and test sets (in npz file)
        """
        """
        Some data have been cached for dev and test data.
        """
        if batch_size == -1:
            cached_data = self.load_set()
            if cached_data:
                return cached_data

        if maxlength is None:
            maxlength = self.maxlength

        if not self.isOpen:
            self.file = open(self.filename, 'r')
            """
            Read first line to remove the header
            """
            self.file.readline()
            self.isOpen = True

        batch_sentence1 = np.empty((0, maxlength), dtype=np.int32)
        batch_length1 = np.empty(0, dtype=np.int32)
        batch_sentence2 = np.empty((0, maxlength), dtype=np.int32)
        batch_length2 = np.empty(0, dtype=np.int32)
        batch_label = np.empty(0, dtype=np.int32)
        """
        If batch_size == -1, then read the whole file
        """
        while batch_size > 0 or batch_size == -1:
            line = self.file.readline()
            if line == '':
                if batch_size == -1:
                    """
                    Finish reading the whole file (for dev and test set)
                    """
                    break
                else:
                    self.epoch += 1
                    """
                    Skip the header line
                    """
                    self.file.seek(0, 0)
                    self.file.readline()
                    continue

            if batch_size != -1:
                batch_size -= 1

            self.total_batch += 1

            line = line.split('|')
            # sentence1 : premise
            s_ids_1 = line[0]

            s_ids_1 = np.fromstring(s_ids_1, dtype=np.int32, sep=' ')
            pad = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0], dtype=np.int32)))
            s_ids_1 = pad(s_ids_1, maxlength)

            batch_sentence1 = np.append(batch_sentence1, [s_ids_1], axis=0)
            s_len_1 = np.int32(line[1]) if np.int32(line[1]) < maxlength else maxlength
            batch_length1 = np.append(batch_length1, s_len_1)

            # sentence2 : hypothesis
            s_ids_2 = line[2]

            s_ids_2 = np.fromstring(s_ids_2, dtype=np.int32, sep=' ')
            pad = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0], dtype=np.int32)))
            s_ids_2 = pad(s_ids_2, maxlength)

            batch_sentence2 = np.append(batch_sentence2, [s_ids_2], axis=0)
            s_len_2 = np.int32(line[3]) if np.int32(line[3]) < maxlength else maxlength
            batch_length2 = np.append(batch_length2, s_len_2)

            # label
            p_label = line[4]
            batch_label = np.append(batch_label, np.int32(p_label))

        if batch_size == -1 and save:
            self.save_set(batch_sentence1, batch_length1, batch_sentence2, batch_length2, batch_label)

        return batch_sentence1, batch_length1, batch_sentence2, batch_length2, batch_label

    def save_set(self, batch_sentence1, batch_length1, batch_sentence2, batch_length2, batch_label, savefile=None):
        if savefile is None:
            savefile = os.path.splitext(os.path.basename(self.filename))[0] + '.npz'
        dir = os.path.dirname(self.filename)
        filepath = os.path.join(dir, savefile)
        with open(filepath, 'wb') as f:
            print('Saving cached data to', filepath)
            np.savez(f, sentence1=batch_sentence1, length1=batch_length1,
                     sentence2=batch_sentence2, length2=batch_length2, label=batch_label)

    def load_set(self, loadfile=None):
        if loadfile is None:
            loadfile = os.path.splitext(os.path.basename(self.filename))[0] + '.npz'
        dir = os.path.dirname(self.filename)
        filepath = os.path.join(dir, loadfile)
        if not os.path.exists(filepath):
            print('No cached file found for', self.filename)
            print('Start reloading file')
            return None
        print('Found cached file for', self.filename)
        with open(filepath, 'rb') as f:
            zfile = np.load(f)
            if zfile['sentence1'][:, :self.maxlength].shape[0] == self.total_num \
                    and zfile['length1'].shape[0] == self.total_num \
                    and zfile['sentence2'][:, :self.maxlength].shape[0] == self.total_num \
                    and zfile['length2'].shape[0] == self.total_num \
                    and zfile['label'].shape[0] == self.total_num:
                print('Verified', filepath)
            return zfile['sentence1'][:, :self.maxlength], zfile['length1'], zfile['sentence2'][:, :self.maxlength], \
                   zfile['length2'], zfile['label']

    def close(self):
        self.epoch = 0
        self.total_batch = 0
        self.isOpen = False
        self.file.close()


class BatchGeneratorFromNumpy(BatchGenerator):
    def __init__(self, filename, maxlength=None):
        """
        :param filename:
        :param np_data: should be a return value from
        :param maxlength:
        :return:
        """
        self.filename = filename
        self.file = None
        self.maxlength = maxlength
        self.np_data = None
        self.epoch = 0
        self.cur_num = 0
        self.total_num = 0
        with open(filename, 'r') as f:
            self.total_num = sum(line.count('\n') for line in BatchGenerator.block(f)) - 1
        print('Found data file:', filename, 'Total number:', self.total_num)

    def load_data(self):
        cached_data = self.load_set()
        if cached_data:
            self.np_data = cached_data
            return
        maxlength = self.maxlength

        wdgts = [pgb.SimpleProgress(), ' ',
             pgb.Bar(marker='∎', left='|', right='|'), ' ',
             pgb.Timer(), ' ',
             pgb.ETA()]

        with open(self.filename, 'r') as f:
            set = BatchGenerator.whole(f).split('\n')

            batch_sentence1 = np.empty((0, maxlength), dtype=np.int32)
            batch_length1 = np.empty(0, dtype=np.int32)
            batch_sentence2 = np.empty((0, maxlength), dtype=np.int32)
            batch_length2 = np.empty(0, dtype=np.int32)
            batch_label = np.empty(0, dtype=np.int32)

            with pgb.ProgressBar(widgets=wdgts, maxval=self.total_num) as p:
                for i, line in enumerate(set):
                    if i == 0:
                        """
                        Skip the first line
                        """
                        continue

                    if not line:
                        break

                    line = line.split('|')
                    # sentence1 : premise
                    s_ids_1 = line[0]

                    s_ids_1 = np.fromstring(s_ids_1, dtype=np.int32, sep=' ')
                    pad = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0], dtype=np.int32)))
                    s_ids_1 = pad(s_ids_1, maxlength)

                    batch_sentence1 = np.append(batch_sentence1, [s_ids_1], axis=0)
                    s_len_1 = np.int32(line[1]) if np.int32(line[1]) < maxlength else maxlength
                    batch_length1 = np.append(batch_length1, s_len_1)

                    # sentence2 : hypothesis
                    s_ids_2 = line[2]

                    s_ids_2 = np.fromstring(s_ids_2, dtype=np.int32, sep=' ')
                    pad = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0], dtype=np.int32)))
                    s_ids_2 = pad(s_ids_2, maxlength)

                    batch_sentence2 = np.append(batch_sentence2, [s_ids_2], axis=0)
                    s_len_2 = np.int32(line[3]) if np.int32(line[3]) < maxlength else maxlength
                    batch_length2 = np.append(batch_length2, s_len_2)

                    # label
                    p_label = line[4]
                    batch_label = np.append(batch_label, np.int32(p_label))

                    p.update(i)

            if batch_sentence1.shape[0] == self.total_num \
                    and batch_length1.shape[0] == self.total_num \
                    and batch_sentence2.shape[0] == self.total_num \
                    and batch_length2.shape[0] == self.total_num \
                    and batch_label.shape[0] == self.total_num:
                print('Verified', self.filename)

            self.save_set(batch_sentence1, batch_length1, batch_sentence2, batch_length2, batch_label)

            self.np_data = (batch_sentence1, batch_length1, batch_sentence2, batch_length2, batch_label)

    def next_batch(self, batch_size, **kwargs):
        if self.np_data is None:
            print('Data not loaded.')
            return None
        maxlength = self.maxlength
        start = self.cur_pointer
        end = self.cur_pointer + batch_size
        if not end < self.total_num:
            end = self.total_num
            self.epoch += 1
            self.cur_pointer = 0
        else:
            self.cur_pointer = end
        self.total_batch += 1
        a, b, c, d, e = self.np_data
        return a[start:end, :self.maxlength], b[start:end], c[start:end, :self.maxlength], \
                   d[start:end], e[start:end]

    def get_epoch(self, batch_size):
        if self.np_data is None:
            print('Data not loaded')
            return
        cur_epoch = self.epoch
        while cur_epoch == self.epoch:
            yield self.next_batch(batch_size)

if __name__ == '__main__':
    import config

    # train_generator = BatchGenerator(config.SNLI_CLEANED_840B_TRAIN_SET_FILE, maxlength=100)
    # dev_generator =BatchGenerator(config.SNLI_CLEANED_840B_DEV_SET_FILE, maxlength=90)
    # data = dev_generator.next_batch(-1)

    train_generator = BatchGeneratorFromNumpy(config.SNLI_CLEANED_840B_TRAIN_SET_FILE, maxlength=85)
    train_generator.load_data()
        # dev_generator = BatchGenerator(config.SNLI_CLEANED_840B_DEV_SET_FILE, maxlength=100)
        # seems pass the test!
        # print(generator.next_batch(3))
