from config import ROOT_DIR, DATA_DIR
from util.save_tool import ResultSaver
from preprocess.sst.batchGenerator import Batch_generator
import os


def wrapper(model_name, model, max_length=80, batch_size=64, benchmark=None):
    p_train_data_path = os.path.join(DATA_DIR, 'Diy/sst/p_train_data.txt')

    s_dev_data_path = os.path.join(DATA_DIR, 'Diy/sst/s_binary_dev_data.txt')
    s_test_data_path = os.path.join(DATA_DIR, 'Diy/sst/s_binary_test_data.txt')

    fine_s_dev_data_path = os.path.join(DATA_DIR, 'Diy/sst/s_dev_data.txt')
    fine_s_test_data_path = os.path.join(DATA_DIR, 'Diy/sst/s_test_data.txt')

    train_generator = Batch_generator(filename=p_train_data_path, maxlength=max_length)

    dev_generator = Batch_generator(filename=s_dev_data_path, maxlength=max_length)
    test_generator = Batch_generator(filename=s_test_data_path, maxlength=max_length)

    dev_data, dev_length, dev_label = dev_generator.next_batch(-1)
    t_data, t_length, t_label = test_generator.next_batch(-1)

    fine_dev_generator = Batch_generator(filename=fine_s_dev_data_path, maxlength=max_length)
    fine_test_generator = Batch_generator(filename=fine_s_test_data_path, maxlength=max_length)

    f_dev_data, f_dev_length, f_dev_label = fine_dev_generator.next_batch(-1)
    f_t_data, f_t_length, f_t_label = fine_test_generator.next_batch(-1)

    # Write to file
    i = 0

    recorder = ResultSaver(model_name=model_name, model=model)
    recorder.setup()

    benchmark_b, benchmark_f = benchmark

    print('Start training.')
    while True:
        data, length, label = train_generator.next_batch(batch_size)
        model.train(data, length, label)

        i += 1

        if i % 100 == 0:
            cur_accuracy = model.predict(dev_data, dev_length, dev_label, name='(dev)')
            test_accuracy = model.predict(t_data, t_length, t_label, name='(test)')

            cur_fine_accuracy = model.predict_fg(f_dev_data, f_dev_length, f_dev_label, name='(dev)')
            test_fine_accuracy = model.predict_fg(f_t_data, f_t_length, f_t_label, name='(test)')

            print('Number of epoch learned:', train_generator.epoch)

            if cur_accuracy > benchmark_b or cur_fine_accuracy > benchmark_f:

                if cur_accuracy > benchmark_b:
                    benchmark_b = cur_accuracy
                if cur_fine_accuracy > benchmark_f:
                    benchmark_f = cur_fine_accuracy

                bin_info = ' '.join(['Dev binary accuracy:',
                                     str(cur_accuracy),
                                     'Test binary accuracy:',
                                     str(test_accuracy)])
                fine_info = ' '.join(['Dev FG accuracy:',
                                      str(cur_fine_accuracy),
                                      'Test FG accuracy:',
                                      str(test_fine_accuracy),
                                      'Number of epoch:',
                                      str(train_generator.epoch)])
                recorder.logging(bin_info, with_time=False)
                recorder.logging(fine_info)

if __name__ == '__main__':
    pass
