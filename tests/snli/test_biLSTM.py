from datetime import datetime
from models.snli.lstm_variant import biLSTM
import config
from util.dict_builder import fine_selected_embedding
from tests.snli.test_wrapper import wrapper
from util.vocab_stat import get_dict
import argparse

if __name__ == '__main__':
    arg_parse = argparse.ArgumentParser(description='Test for some variant of LSTM model.')
    arg_parse.add_argument('-m', '--message', metavar='message', dest='message', action='store',
                           help='Specify a message for this test')

    args = arg_parse.parse_args()
    
    max_length = 80

    # 2196018
    # 400001
    voc_size = 2196018

    dict_ = get_dict()
    word2id, word_embedding = fine_selected_embedding(config.GLOVE_840B_PATH, dict_, pre_vocab_size=voc_size)

    timestamp = '{0:(%Y-%m-%d-%H:%M:%S)}'.format(datetime.now())

    biLSTM = biLSTM(lstm_step=max_length, embedding=word_embedding, hidden_d=100, vocab_size=voc_size, Time=timestamp,
                    Model_Name='BiLSTM', Message=args.message)
    biLSTM.setup(embedding=word_embedding)
    wrapper(model_name='snli-biLSTM', model=biLSTM, max_length=max_length, benchmark=0.70)
