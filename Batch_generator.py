import numpy as np


class Batch_generator:
    def __init__(self, filename, maxlength=None):
        self.filename = filename
        self.file = None
        self.maxlength = maxlength
        self.isOpen = False
        self.epoch = 0
        self.total_size = 0

    def next_batch(self, batch_size, maxlength=None):
        '''
        truncate the length to maxlength
        '''
        if maxlength is None:
            maxlength = self.maxlength

        if not self.isOpen:
            self.file = open(self.filename, 'r')
            self.isOpen = True

        batch_data = np.empty((0, maxlength), dtype=np.int32)
        batch_length = np.empty(0, dtype=np.int32)
        batch_label = np.empty(0, dtype=np.float32)

        # if batch_size == -1, then read the whole file
        while batch_size > 0 or batch_size == -1:
            line = self.file.readline()
            if line == '':
                if batch_size == -1:
                    break
                else:
                    self.epoch += 1
                    self.file.seek(0, 0)
                    continue

            if batch_size != -1:
                batch_size -= 1

            self.total_size += 1

            line = line.split('|')
            s_ids = line[0]

            s_ids = np.fromstring(s_ids, dtype=np.int32, sep=' ')
            pad = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0], dtype=np.int32)))
            s_ids = pad(s_ids, maxlength)

            batch_data = np.append(batch_data, [s_ids], axis=0)
            s_len = line[1]
            batch_length = np.append(batch_length, np.int32(s_len))
            s_sentiment = line[2]
            batch_label = np.append(batch_label, np.float32(s_sentiment))

        return batch_data, batch_length, batch_label

    def close(self):
        self.epoch = 0
        self.total_size = 0
        self.isOpen = False
        self.file.close()


generator = Batch_generator(filename='/Users/Eason/RA/landOfflol/datasets/Diy/sst/p_train_data.txt', maxlength=30)

data, length, label = generator.next_batch(512)

print(data)
print(length)
print(label)