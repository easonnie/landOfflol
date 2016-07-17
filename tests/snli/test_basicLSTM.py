from datetime import datetime
from models.snli.baselineModels_snli import SnliBasicLSTM
from config import DATA_DIR
from util.dict_builder import fine_selected_embedding
from tests.snli.test_wrapper import wrapper
from util.vocab_stat import get_dict
import os

if __name__ == '__main__':

    max_length = 80

    # 2196018
    # 400001
    voc_size = 2196018

    dict_ = get_dict()

    file_path = os.path.join(DATA_DIR, 'Glove/glove.840B.300d.zip')
    word2id, word_embedding = fine_selected_embedding(file_path, dict_, pre_vocab_size=voc_size)

    timestamp = '{0:(%Y-%m-%d-%H:%M:%S)}'.format(datetime.now())

    basicLSTM = SnliBasicLSTM(lstm_step=max_length, embedding=word_embedding, hidden_d=100, vocab_size=voc_size, Time=timestamp)
    basicLSTM.setup()
    basicLSTM.load_embedding(word_embedding)
    wrapper(model_name='snli-basicLSTM', model=basicLSTM, max_length=max_length, benchmark=0.70)