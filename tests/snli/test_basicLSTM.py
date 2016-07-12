from preprocess.snli.batch_generator_snli import BatchGenerator
from models.snli.baselineModels_snli import SnliBasicLSTM
from config import ROOT_DIR, DATA_DIR
import os

if __name__ == '__main__':

    max_length = 50
    basicLSTM = SnliBasicLSTM(lstm_step=max_length)
    dev_batch_generator = BatchGenerator(os.path.join(DATA_DIR, 'Diy/snli/dev_data.txt'), maxlength=max_length)
    train_batch_generator = BatchGenerator(os.path.join(DATA_DIR, '/Diy/snli/train_data.txt'), maxlength=max_length)

    dev_premise, dev_premise_len, dev_hypothesis, dev_hypothesis_len, dev_label = dev_batch_generator.next_batch(-1)

    in_dev_premise = basicLSTM.input_loader.raw_premise
    in_dev_premise_len = basicLSTM.input_loader.premise_length
    in_dev_hypothesis = basicLSTM.input_loader.raw_hypothesis
    in_dev_hypothesis_len = basicLSTM.input_loader.hypothesis_length
    in_dev_label = basicLSTM.input_loader.label

    in_dev_feed_dit = {in_dev_premise: dev_premise,
                       in_dev_premise_len: dev_premise_len,
                       in_dev_hypothesis: dev_hypothesis,
                       in_dev_hypothesis_len: dev_hypothesis_len,
                       in_dev_label: dev_label}

    i = 0
    basicLSTM.setup()

    while True:
        premise, premise_len, hypothesis, hypothesis_len, label = train_batch_generator.next_batch(128)

        in_premise = basicLSTM.input_loader.raw_premise
        in_premise_len = basicLSTM.input_loader.premise_length
        in_hypothesis = basicLSTM.input_loader.raw_hypothesis
        in_hypothesis_len = basicLSTM.input_loader.hypothesis_length
        in_label = basicLSTM.input_loader.label

        in_feed_dit = {in_premise: premise,
                       in_premise_len: premise_len,
                       in_hypothesis: hypothesis,
                       in_hypothesis_len: hypothesis_len,
                       in_label: label}

        basicLSTM.train(feed_dict=in_feed_dit)
        i += 1

        if i % 100 == 0:
            print('dev:')
            basicLSTM.predict(feed_dict=in_dev_feed_dit)
            print('train:')
            basicLSTM.predict(feed_dict=in_feed_dit)
            print('Number of epoch:', train_batch_generator.epoch)
