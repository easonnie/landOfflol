from preprocess.snli.batch_generator_snli import BatchGenerator
from config import ROOT_DIR, DATA_DIR
import os


def show_stat(ac, cost, context):
    print(context, 'accuracy:', ac)
    print(context, 'cost:', cost)


def wrapper(model, max_length=60, benchmark=None):

    print('Loading data.')
    dev_batch_generator = BatchGenerator(os.path.join(DATA_DIR, 'Diy/snli/dev_data.txt'), maxlength=max_length)
    train_batch_generator = BatchGenerator(os.path.join(DATA_DIR, '/Diy/snli/train_data.txt'), maxlength=max_length)

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
    model.setup()

    print('Start training.')
    while True:
        premise, premise_len, hypothesis, hypothesis_len, label = train_batch_generator.next_batch(128)

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
            show_stat(train_acc, train_acc, 'Train')

            print('Number of epoch:', train_batch_generator.epoch)