from preprocess.snli.batch_generator_snli import BatchGeneratorH5
from util.save_tool import ResultSaver
import os
import config
import datetime
import numpy as np


def show_stat(ac, cost, context):
    print(context, 'accuracy:', ac)
    print(context, 'cost:', cost)


def wrapper(model_name, model, batch_size=128, max_length=80, benchmark=None):

    TRAIN_FILE = config.SNLI_840B_H5_TRAIN_FILE
    TEST_FILE = config.SNLI_840B_H5_TEST_FILE
    DEV_FILE = config.SNLI_840B_H5_DEV_FILE

    print('Loading data from', TRAIN_FILE)
    dev_batch_generator = BatchGeneratorH5(DEV_FILE, maxlength=max_length)
    test_batch_generator = BatchGeneratorH5(TEST_FILE, maxlength=max_length)
    train_batch_generator = BatchGeneratorH5(TRAIN_FILE, maxlength=max_length)

    dev_batch_generator.load_data()
    test_batch_generator.load_data()
    train_batch_generator.load_data()

    dev_data = dev_batch_generator.next_batch(-1)
    test_data = test_batch_generator.next_batch(-1)

    dev_input_dict = model.input_loader.feed_dict_builder(dev_data)
    test_input_dict = model.input_loader.feed_dict_builder(test_data)

    i = 0
    recorder = ResultSaver(model_name=model_name, model=model)
    recorder.setup()

    # BENCHMARK Important
    if benchmark is None:
        benchmark = 0.70

    print('Start training.')
    start = datetime.datetime.now()

    for i in range(15):
        best_acc = 0
        less_cost = 1
        train_acc_list = []
        train_cost_list = []
        print('Number of epoch:', i)
        for j, cur_batch_data in enumerate(train_batch_generator.get_epoch(batch_size=batch_size)):
            cur_train_batch_dict = model.input_loader.feed_dict_builder(cur_batch_data)
            model.train(feed_dict=cur_train_batch_dict)
            train_acc, train_cost = model.predict(feed_dict=cur_train_batch_dict)

            train_acc_list.append(train_acc)
            train_cost_list.append(train_cost)

            if j % 100 == 0:
                dev_acc, dev_cost = model.predict(feed_dict=dev_input_dict)
                show_stat(dev_acc, dev_cost, 'Dev')

                best_acc = dev_acc if dev_acc > best_acc else best_acc
                less_cost = dev_cost if dev_cost < less_cost else less_cost

                if dev_acc > benchmark:
                    benchmark = dev_acc

                    test_acc, test_cost = model.predict(feed_dict=test_input_dict)
                    show_stat(test_acc, test_cost, 'Test')

                    info = ' '.join(['Dev accuracy:', str(dev_acc),
                                     'Train accuracy:', str(train_acc),
                                     'Test accuracy:', str(test_acc),
                                     'Number of epoch:',
                                     str(train_batch_generator.epoch)])
                    recorder.logging(info)

                print('Current epoch:', train_batch_generator.epoch)
                print('Time consumed:', str(datetime.datetime.now() - start))

        print('Epoch', i, 'finished')
        print('Stats:')
        print('Best dev accuracy:', best_acc, 'Less cost:', less_cost)
        print('Average train accuracy', np.mean(train_acc_list))
        print('Average train cost', np.mean(train_cost_list))
        msg = ' - '.join(['epoch:', str(i),
                          'best dev acc:', str(best_acc),
                          'least cost', str(less_cost),
                          'avg train acc', str(np.mean(train_acc_list)),
                          'avg train cost', str(np.mean(train_cost_list))
                          ])

        recorder.logging_epoch(msg)


# def wrapper(model_name, model, max_length=80, batch_size=128, benchmark=None):
#
#     print('Loading data from', os.path.join(config.SNLI_CLEANED_840B_TRAIN_SET_FILE))
#     dev_batch_generator = BatchGenerator(os.path.join(config.SNLI_CLEANED_840B_DEV_SET_FILE), maxlength=max_length)
#     test_batch_generator = BatchGenerator(os.path.join(config.SNLI_CLEANED_840B_TEST_SET_FILE), maxlength=max_length)
#     train_batch_generator = BatchGenerator(os.path.join(config.SNLI_CLEANED_840B_TRAIN_SET_FILE), maxlength=max_length)
#
#     dev_data = dev_batch_generator.next_batch(-1)
#     test_data = test_batch_generator.next_batch(-1)
#
#     dev_input_dict = model.input_loader.feed_dict_builder(dev_data)
#     test_input_dict = model.input_loader.feed_dict_builder(test_data)
#
#     i = 0
#     recorder = ResultSaver(model_name=model_name, model=model)
#     recorder.setup()
#
#     # BENCHMARK Important
#     if benchmark is None:
#         benchmark = 0.70
#
#     print('Start training.')
#     start = datetime.datetime.now()
#     while True:
#         cur_batch_data = train_batch_generator.next_batch(batch_size)
#
#         cur_train_batch_dict = model.input_loader.feed_dict_builder(cur_batch_data)
#
#         model.train(feed_dict=cur_train_batch_dict)
#         i += 1
#
#         if i % 100 == 0:
#             dev_acc, dev_cost = model.predict(feed_dict=dev_input_dict)
#             show_stat(dev_acc, dev_cost, 'Dev')
#
#             train_acc, train_cost = model.predict(feed_dict=cur_train_batch_dict)
#             show_stat(train_acc, train_cost, 'Train')
#
#             if dev_acc > benchmark:
#                 benchmark = dev_acc
#
#                 test_acc, test_cost = model.predict(feed_dict=test_input_dict)
#                 show_stat(test_acc, test_cost, 'Test')
#
#                 info = ' '.join(['Dev accuracy:', str(dev_acc),
#                                  'Train accuracy:', str(train_acc),
#                                  'Test accuracy:', str(test_acc),
#                                  'Number of epoch:',
#                                  str(train_batch_generator.epoch)])
#                 recorder.logging(info)
#
#             print('Number of epoch:', train_batch_generator.epoch)
#             print('Time consumed:', str(datetime.datetime.now() - start))

            # asctime  ref: https://docs.python.org/3/howto/logging.html
