from preprocess.snli.batch_generator_snli import BatchGenerator
from config import ROOT_DIR, DATA_DIR
from util.save_tool import ResultSaver
import os
import sys
import config


def show_stat(ac, cost, context):
    print(context, 'accuracy:', ac)
    print(context, 'cost:', cost)


def wrapper(model_name, model, max_length=80, batch_size=128, benchmark=None):

    print('Loading data from', os.path.join(config.SNLI_CLEANED_840B_TRAIN_SET_FILE))
    dev_batch_generator = BatchGenerator(os.path.join(config.SNLI_CLEANED_840B_DEV_SET_FILE), maxlength=max_length)
    test_batch_generator = BatchGenerator(os.path.join(config.SNLI_CLEANED_840B_TEST_SET_FILE), maxlength=max_length)
    train_batch_generator = BatchGenerator(os.path.join(config.SNLI_CLEANED_840B_TRAIN_SET_FILE), maxlength=max_length)

    #TODO build feed_dict for data input

    dev_premise, dev_premise_len, dev_hypothesis, dev_hypothesis_len, dev_label = dev_batch_generator.next_batch(-1)

    in_dev_premise = model.input_loader.raw_premise
    in_dev_premise_len = model.input_loader.premise_length
    in_dev_hypothesis = model.input_loader.raw_hypothesis
    in_dev_hypothesis_len = model.input_loader.hypothesis_length
    in_dev_label = model.input_loader.label

    in_dev_feed_dit = {in_dev_premise: dev_premise,
                       in_dev_premise_len: dev_premise_len,
                       in_dev_hypothesis: dev_hypothesis,
                       in_dev_hypothesis_len: dev_hypothesis_len,
                       in_dev_label: dev_label}

    i = 0
    recorder = ResultSaver(model_name=model_name, model=model)
    recorder.setup()
    # BENCHMARK Important
    if benchmark is None:
        benchmark = 0.70

    print('Start training.')
    while True:
        premise, premise_len, hypothesis, hypothesis_len, label = train_batch_generator.next_batch(batch_size)

        in_premise = model.input_loader.raw_premise
        in_premise_len = model.input_loader.premise_length
        in_hypothesis = model.input_loader.raw_hypothesis
        in_hypothesis_len = model.input_loader.hypothesis_length
        in_label = model.input_loader.label

        in_feed_dit = {in_premise: premise,
                       in_premise_len: premise_len,
                       in_hypothesis: hypothesis,
                       in_hypothesis_len: hypothesis_len,
                       in_label: label}

        model.train(feed_dict=in_feed_dit)
        i += 1

        if i % 100 == 0:
            dev_acc, dev_cost = model.predict(feed_dict=in_dev_feed_dit)
            show_stat(dev_acc, dev_cost, 'Dev')

            train_acc, train_cost = model.predict(feed_dict=in_feed_dit)
            show_stat(train_acc, train_cost, 'Train')

            if dev_acc > benchmark:
                benchmark = dev_acc
                info = ' '.join(['Dev accuracy:', str(dev_acc), 'Train accuracy:', str(train_acc), 'Number of epoch:',
                                 str(train_batch_generator.epoch)])
                recorder.logging(info)

            print('Number of epoch:', train_batch_generator.epoch)
            sys.stdout.flush()