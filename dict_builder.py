import numpy as np
import pandas as pd

# em = pd.read_csv(path + 'datasets/Glove/glove.6B.300d.txt', sep=' ', header=None, quoting=3, encoding='utf-8')


def voc_builder(vocab_path):
    em_df = pd.read_csv(vocab_path, sep=' ', header=None, quoting=3, encoding='utf-8')
    word_dict = dict()
    for i, row in em_df.iterrows():
        word_dict[row[0]] = i

    word_dict['<unk>'] = len(word_dict)

    embedding = np.asarray(em_df.iloc[:, 1:])
    embedding = np.row_stack((embedding, embedding.mean(axis=0)))

    return word_dict, embedding


if __name__ == '__main__':
    path = '/Users/Eason/RA/landOfflol/'
    word2id, word_embedding = voc_builder(path + 'datasets/Glove/glove.6B.300d.txt')
    print(len(word2id))
