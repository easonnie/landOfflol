import numpy as np
import pandas as pd
import os

path = 'datasets/SST/'


def raw_reader(path):
    phase_map_df = pd.read_csv(path + 'dictionary.txt' , sep='|', quoting=3, encoding='utf-8')
    sentiment_labels_df = pd.read_csv(path + 'sentiment_labels.txt', sep='|', quoting=3, encoding='utf-8')

    phases_with_labels = phase_map_df.merge(sentiment_labels_df, on='phrase ids')
    phase_map_df.sort_index(inplace=True)

    sentences_df = pd.read_csv(path + 'datasetSentences.txt', sep='\t', quoting=3, encoding='utf-8')
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

print(phase_df)

# phase_map_df = pd.read_csv('datasets/SST/dictionary.txt', sep='|', quoting=3, encoding='utf-8')
# sentiment_labels_df = pd.read_csv('datasets/SST/sentiment_labels.txt', sep='|', quoting=3, encoding='utf-8')
#
# phases_with_labels = phase_map_df.merge(sentiment_labels_df, on='phrase ids')
# # print(phases_with_labels)
#
# # sort the index
# phase_map_df.sort_index(inplace=True)
#
#
# sentences_df = pd.read_csv('datasets/SST/datasetSentences.txt', sep='\t', quoting=3, encoding='utf-8')
# data_split_df = pd.read_csv('datasets/SST/datasetSplit.txt', sep=',', quoting=3, encoding='utf-8')
#
# sentences_df = sentences_df.merge(data_split_df, on='sentence_index')
# # print(sentences_df)
# sentences_with_labels = sentences_df.merge(phases_with_labels, on='sentence', how='inner')
# # print(sentences_with_labels)
#
# train_df = sentences_with_labels[sentences_with_labels.ix[:, 'splitset_label'] == 1]
# print(train_df.shape)
#
# test_df = sentences_with_labels[sentences_with_labels.ix[:, 'splitset_label'] == 2]
# print(test_df.shape)
#
# dev_df = sentences_with_labels[sentences_with_labels.ix[:, 'splitset_label'] == 3]
# print(dev_df.shape)
#
# print(phases_with_labels)
