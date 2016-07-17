import numpy as np
import pandas as pd

from etc.preprocessor import build_new_sets
from util.dict_builder import build_voc, fine_selected_embedding
from util.vocab_stat import get_dict
from config import DATA_DIR
import os

phrase_train_df = pd.read_csv('/Users/Eason/RA/landOfflol/datasets/Diy/sst/intermediates/phase_train.txt', sep='|',
                              quoting=3, encoding='utf-8')
# 3185582
# print(phrase_train_df.dropna(axis=0))
phase_map_df = pd.read_csv(DATA_DIR + '/SST/dictionary.txt', sep='|', quoting=3, encoding='utf-8')
# 239242
sentiment_labels_df = pd.read_csv(DATA_DIR + '/SST/sentiment_labels.txt', sep='|', quoting=3, encoding='utf-8')

phases_with_labels = phase_map_df.merge(sentiment_labels_df, on='phrase ids')
phrase_train_df = phrase_train_df.merge(phases_with_labels, on='sentence')

phrase_train_df = phrase_train_df.reindex(np.random.permutation(phrase_train_df.index))

lower_df = phrase_train_df['sentence'].map(lambda x: x.lower())
phrase_train_df['sentence'] = lower_df.drop_duplicates()
phrase_train_df = phrase_train_df.dropna(axis=0)

fine_dict = get_dict()
word2id, word_embedding = fine_selected_embedding(os.path.join(DATA_DIR, 'Glove/glove.840B.300d.zip'),
                                                  fine_dict, pre_vocab_size=2196018)

build_new_sets(phrase_train_df, word2id, os.path.join(DATA_DIR, 'Diy/sst/fs/p_train_data.txt'))
