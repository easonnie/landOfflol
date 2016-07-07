import numpy as np
import pandas as pd
import os

path = 'datasets/SST/'


def shuffler(filename):
  df = pd.read_csv(filename, header=0)
  # return the pandas dataframe
  return df.reindex(np.random.permutation(df.index))


def raw_reader(path):
    phase_map_df = pd.read_csv(path + 'dictionary.txt' , sep='|', quoting=3, encoding='utf-8')
    sentiment_labels_df = pd.read_csv(path + 'sentiment_labels.txt', sep='|', quoting=3, encoding='utf-8')

    phases_with_labels = phase_map_df.merge(sentiment_labels_df, on='phrase ids')
    phase_map_df.sort_index(inplace=True)

    sentences_df = pd.read_csv(path + 'datasetSentences-cleaned.txt', sep='\t', quoting=3, encoding='iso-8859-1')
    data_split_df = pd.read_csv(path + 'datasetSplit.txt', sep=',', quoting=3, encoding='utf-8')

    sentences_df = sentences_df.merge(data_split_df, on='sentence_index')
    sentences_with_labels = sentences_df.merge(phases_with_labels, on='sentence', how='inner')

    train_df = sentences_with_labels[sentences_with_labels.ix[:, 'splitset_label'] == 1]
    test_df = sentences_with_labels[sentences_with_labels.ix[:, 'splitset_label'] == 2]
    dev_df = sentences_with_labels[sentences_with_labels.ix[:, 'splitset_label'] == 3]

    mask = phases_with_labels['sentence'].isin(test_df['sentence'])
    cleaned_phase_df = phases_with_labels[~mask]
    mask = cleaned_phase_df['sentence'].isin(dev_df['sentence'])
    cleaned_phase_df = cleaned_phase_df[~mask]

    return cleaned_phase_df, train_df, dev_df, test_df

phase_df, train_df, dev_df, test_df = raw_reader(path)
phase_df = phase_df.reindex(np.random.permutation(phase_df.index))
# print(phase_df)


from dict_builder import voc_builder

path = '/Users/Eason/RA/landOfflol/'
word2id, word_embedding = voc_builder(path + 'datasets/Glove/glove.6B.300d.txt')


def build_new_sets(df, dict, path=None, isBinary=False):
    file = open(path, 'w')

    for i, row in df.iterrows():
        sentence_words = row['sentence'].lower().split()

        # Change unseen words to '<unk>'
        for j in range(len(sentence_words)):
            if sentence_words[j] not in dict:
                sentence_words[j] = '<unk>'

        sentiment_value = row['sentiment values']
        if isBinary is True and 0.4 < sentiment_value <= 0.6:
            continue;

        sentence_ids = [dict[word] for word in sentence_words]
        length = len(sentence_ids)

        print(sentence_ids, length, sentiment_value)

        file.write(' '.join([str(_id) for _id in sentence_ids]))
        file.write('|')
        file.write(str(length))
        file.write('|')
        file.write(str(sentiment_value))
        file.write('\n')

    file.close()

# build_new_sets(phase_df, word2id, path='/Users/Eason/RA/landOfflol/datasets/Diy/sst/p_train_data.txt')
# build_new_sets(train_df, word2id, path='/Users/Eason/RA/landOfflol/datasets/Diy/sst/s_train_data.txt')
# build_new_sets(dev_df, word2id, path='/Users/Eason/RA/landOfflol/datasets/Diy/sst/s_dev_data.txt')
# build_new_sets(test_df, word2id, path='/Users/Eason/RA/landOfflol/datasets/Diy/sst/s_test_data.txt')
build_new_sets(test_df, word2id, path='/Users/Eason/RA/landOfflol/datasets/Diy/sst/s_binary_test_data.txt', isBinary=True)