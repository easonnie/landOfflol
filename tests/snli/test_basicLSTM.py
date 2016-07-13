from datetime import datetime
from models.snli.baselineModels_snli import SnliBasicLSTM
from config import ROOT_DIR, DATA_DIR
from util.dict_builder import voc_builder
from tests.snli.test_wrapper import wrapper
import os

if __name__ == '__main__':

    max_length = 70

    print('Loading word embedding.')
    word2id, word_embedding = voc_builder(os.path.join(DATA_DIR, 'Glove/glove.6B.300d.txt'))

    timestamp = '{0:(%Y-%m-%d-%H:%M:%S)}'.format(datetime.now())

    basicLSTM = SnliBasicLSTM(lstm_step=max_length, embedding=word_embedding, hidden_d=150, Time=timestamp)

    wrapper(model_name='snli-basicLSTM', model=basicLSTM, max_length=max_length, benchmark=0.76)