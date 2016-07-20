import h5py


class SkipthBatchGenerator:
    def __init__(self, filename):
        """
        :param filename:
        :param np_data: should be a return value from
        :param maxlength:
        :return:
        """
        self.filename = filename
        self.file = None
        self.np_data = None
        self.epoch = 0
        self.cur_num = 0
        self.total_num = 0
        self.total_batch = 0
        self.cur_pointer = 0

    def load_data(self):
        self.file = h5py.File(self.filename, "r")
        self.np_data = self.file['premise'], self.file['hypothesis'], self.file['label']
        self.total_num = self.file['premise'].shape[0]
        if self.file['premise'].shape[0] == self.file['hypothesis'].shape[0] == self.file['label'].shape[0]:
            print(self.filename, 'verified')
        return self.total_num # return total number of data

    def next_batch(self, batch_size, **kwargs):
        if self.np_data is None:
            print('Data not loaded.')
            return None

        if batch_size == -1:
            premise, hypothesis, label = self.np_data
            return premise[:], hypothesis[:], label[:]

        start = self.cur_pointer
        end = self.cur_pointer + batch_size
        if not end < self.total_num:
            end = self.total_num
            self.epoch += 1
            self.cur_pointer = 0
        else:
            self.cur_pointer = end
        self.total_batch += 1
        premise, hypothesis, label = self.np_data
        return premise[start:end], hypothesis[start:end], label[start:end]

    def get_epoch(self, batch_size):
        if self.np_data is None:
            print('Data not loaded')
            return
        cur_epoch = self.epoch
        while cur_epoch == self.epoch:
            yield self.next_batch(batch_size)


if __name__ == '__main__':
    import config
    from preprocess.snli.batch_generator_snli import BatchGenerator
    import numpy as np

    # test_generator_ = BatchGenerator(config.SNLI_CLEANED_840B_TEST_SET_FILE, maxlength=85)
    # *a, c_ = test_generator_.next_batch(-1)
    #
    # test_generator = SkipthBatchGenerator(config.SNLI_SKIPTH_TEST_FILE_ON_MAC)
    # test_generator.load_data()
    #
    # a, b, c = test_generator.next_batch(-1)
    # print(np.all(c == c_))
    #
    # dev_generator_ = BatchGenerator(config.SNLI_CLEANED_840B_DEV_SET_FILE, maxlength=85)
    # *a, c_ = dev_generator_.next_batch(-1)
    #
    # dev_generator = SkipthBatchGenerator(config.SNLI_SKIPTH_DEV_FILE_ON_MAC)
    # dev_generator.load_data()
    #
    # for i, dev_data in enumerate(dev_generator.get_epoch(batch_size=2560)):
    #     dev_data_ = dev_generator_.next_batch(batch_size=2560)
    #     *a, c_ = dev_data_
    #     *a, c = dev_data
    #     print(i * 2560, (i + 1) * 2560)
    #     print(i * 2560, i * 2560 + len(c))
    #     print(np.all(c == c_))
    train_generator_ = BatchGenerator(config.SNLI_CLEANED_840B_TRAIN_SET_FILE, maxlength=85)

    train_generator = SkipthBatchGenerator(config.SNLI_SKIPTH_TRAIN_FILE_ON_MAC)
    train_generator.load_data()

    for i, dev_data in enumerate(train_generator.get_epoch(batch_size=2560)):
        dev_data_ = train_generator_.next_batch(batch_size=2560)
        *a, c_ = dev_data_
        *a, c = dev_data
        print(i * 2560, (i + 1) * 2560)
        print(i * 2560, i * 2560 + len(c))
        ver = np.all(c == c_)
        print(ver)
        if not ver:
            print('Inconsistence')
