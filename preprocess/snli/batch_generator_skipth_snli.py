class SkipthBatchGenerator:
    def __init__(self, filename):
        """
        :param filename:
        :param np_data: should be a return value from
        :param maxlength:
        :return:
        """
        self.filename = filename
        self.np_data = None
        self.epoch = 0
        self.cur_num = 0
        self.total_num = 0
        self.total_batch = 0
        self.cur_pointer = 0

    def load_data(self):
        # TODO load data
        pass
        self.np_data = None
        self.total_num = None
        return self.total_num # return total number of data

    def next_batch(self, batch_size, **kwargs):
        if self.np_data is None:
            print('Data not loaded.')
            return None

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
    pass
# TODO something here