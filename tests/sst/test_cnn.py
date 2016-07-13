from datetime import datetime
from models.sst.cnnText import CNNText
from config import ROOT_DIR, DATA_DIR
from util.dict_builder import voc_builder
from tests.sst.test_wrapper import wrapper
import os

if __name__ == '__main__':

    max_length = 80

    print('Loading word embedding.')
    word2id, word_embedding = voc_builder(os.path.join(DATA_DIR, 'Glove/glove.6B.300d.txt'))

    timestamp = '{0:(%Y-%m-%d-%H:%M:%S)}'.format(datetime.now())

    cnn_model = CNNText(word_embedding, Time=timestamp)

    wrapper(model_name='sst-cnn-nonstatic', model=cnn_model, max_length=max_length, batch_size=64, benchmark=(0.00, 0.00))