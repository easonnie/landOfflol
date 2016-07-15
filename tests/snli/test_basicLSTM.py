from datetime import datetime
from models.snli.baselineModels_snli import SnliBasicLSTM
from config import ROOT_DIR, DATA_DIR
from util.dict_builder import build_voc
from tests.snli.test_wrapper import wrapper
import os

if __name__ == '__main__':

    max_length = 70

    # 2196018
    # 400001
    voc_size = 2196018

    file_path = os.path.join(DATA_DIR, 'Glove/glove.840B.300d.zip')
    word2id, word_embedding = build_voc(file_path,  voc_size)

    timestamp = '{0:(%Y-%m-%d-%H:%M:%S)}'.format(datetime.now())

    basicLSTM = SnliBasicLSTM(lstm_step=max_length, embedding=word_embedding, hidden_d=100, vocab_size=voc_size, Time=timestamp)
    basicLSTM.setup()
    basicLSTM.load_embedding(word_embedding)
    wrapper(model_name='snli-basicLSTM', model=basicLSTM, max_length=max_length, benchmark=0.76)