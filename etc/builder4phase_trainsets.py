import numpy as np
import pandas as pd

from etc.preprocessor import build_new_sets
from util.dict_builder import voc_builder

path = '/Users/Eason/RA/landOfflol/datasets/'

phrase_train_df = pd.read_csv('/Users/Eason/RA/landOfflol/datasets/Diy/sst/intermediates/phase_train.txt', sep='|', quoting=3, encoding='utf-8')
# 3185582
# print(phrase_train_df.dropna(axis=0))
phase_map_df = pd.read_csv(path + 'SST/dictionary.txt', sep='|', quoting=3, encoding='utf-8')
# 239242
sentiment_labels_df = pd.read_csv(path + 'SST/sentiment_labels.txt', sep='|', quoting=3, encoding='utf-8')

phases_with_labels = phase_map_df.merge(sentiment_labels_df, on='phrase ids')
phrase_train_df = phrase_train_df.merge(phases_with_labels, on='sentence')

phrase_train_df = phrase_train_df.reindex(np.random.permutation(phrase_train_df.index))

lower_df = phrase_train_df['sentence'].map(lambda x: x.lower())
phrase_train_df['sentence'] = lower_df.drop_duplicates()
phrase_train_df = phrase_train_df.dropna(axis=0)


path = '/Users/Eason/RA/landOfflol/'
word2id, word_embedding = voc_builder(path + 'datasets/Glove/glove.6B.300d.txt')

build_new_sets(phrase_train_df, word2id, path='/Users/Eason/RA/landOfflol/datasets/Diy/sst/p_train_data.txt')