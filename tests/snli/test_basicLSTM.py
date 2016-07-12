from models.snli.baselineModels_snli import SnliBasicLSTM
from config import ROOT_DIR, DATA_DIR
from util.dict_builder import voc_builder
from .test_wrapper import wrapper
import os

if __name__ == '__main__':

    max_length = 60

    print('Loading word embedding.')
    word2id, word_embedding = voc_builder(os.path.join(DATA_DIR, '/Glove/glove.6B.300d.txt'))
    basicLSTM = SnliBasicLSTM(lstm_step=max_length, embedding=word_embedding)

    wrapper(basicLSTM, max_length=max_length)


