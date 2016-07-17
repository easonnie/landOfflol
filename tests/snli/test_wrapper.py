from preprocess.snli.batch_generator_snli import BatchGenerator
from util.save_tool import ResultSaver
import os
import config
import datetime


def show_stat(ac, cost, context):
    print(context, 'accuracy:', ac)
    print(context, 'cost:', cost)


def wrapper(model_name, model, max_length=80, batch_size=128, benchmark=None):

    print('Loading data from', os.path.join(config.SNLI_CLEANED_840B_TRAIN_SET_FILE))
    dev_batch_generator = BatchGenerator(os.path.join(config.SNLI_CLEANED_840B_DEV_SET_FILE), maxlength=max_length)
    test_batch_generator = BatchGenerator(os.path.join(config.SNLI_CLEANED_840B_TEST_SET_FILE), maxlength=max_length)
    train_batch_generator = BatchGenerator(os.path.join(config.SNLI_CLEANED_840B_TRAIN_SET_FILE), maxlength=max_length)

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
    while True:
        cur_batch_data = train_batch_generator.next_batch(batch_size)

        cur_train_batch_dict = model.input_loader.feed_dict_builder(cur_batch_data)

        model.train(feed_dict=cur_train_batch_dict)
        i += 1

        if i % 100 == 0:
            dev_acc, dev_cost = model.predict(feed_dict=dev_input_dict)
            show_stat(dev_acc, dev_cost, 'Dev')

            train_acc, train_cost = model.predict(feed_dict=cur_train_batch_dict)
            show_stat(train_acc, train_cost, 'Train')

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

            print('Number of epoch:', train_batch_generator.epoch)
            print('Time consumed:', str(datetime.datetime.now() - start))

            # asctime  ref: https://docs.python.org/3/howto/logging.html
