from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import progressbar as pgb
import zipfile
import pickle
from os.path import basename
import os
from pprint import pprint


def voc_builder(vocab_path):
    em_df = pd.read_csv(vocab_path, sep=' ', header=None, quoting=3, encoding='utf-8')
    word_dict = dict()
    for i, row in em_df.iterrows():
        word_dict[row[0]] = i

    word_dict['<unk>'] = len(word_dict)

    embedding = np.asarray(em_df.iloc[:, 1:])
    embedding = np.row_stack((embedding, embedding.mean(axis=0)))

    return word_dict, embedding


def build_voc(vocab_path, pre_vocab_size, word_d=300, name=None, chunksize=120000):

    if name is None:
        name = os.path.splitext(basename(vocab_path))[0]
    dir = os.path.dirname(vocab_path)

    word_dict, embedding = load_embedding(dir, name)

    if word_dict is not None and embedding is not None:
        return word_dict, embedding

    word_dict = {}
    embedding = np.empty((0, word_d))

    em_zipfile = zipfile.ZipFile(vocab_path)
    em_df = pd.read_csv(em_zipfile.open(em_zipfile.namelist()[0], 'r'),
                        sep=' ', header=None, quoting=3, encoding='utf-8',
                        keep_default_na=False, iterator=True, chunksize=chunksize)

    print('Approximate', pre_vocab_size, 'words')
    wdgts = [pgb.SimpleProgress(), ' ',
             pgb.Bar(marker='∎', left='|', right='|'), ' ',
             pgb.Timer(), ' ',
             pgb.ETA()]

    with pgb.ProgressBar(widgets=wdgts, maxval=pre_vocab_size) as p:
        for i, chuck in enumerate(em_df):
            for j, row in chuck.iterrows():
                if row[0] in word_dict:
                    # print('line:(', i * chunksize + j, word_dict[row[0]], ')')
                    # print('word:', row[0])
                    print('Ignore some unknown words')
                    continue
                word_dict[row[0]] = i * chunksize + j
                p.update(i * chunksize + j)
            embedding = np.row_stack((embedding, np.asarray(chuck.iloc[:, 1:])))

    embedding = np.row_stack((embedding, embedding.mean(axis=0)))
    word_dict['<unk>'] = len(word_dict)
    # print(len(word_dict))
    # print(embedding.shape)
    # embedding = np.asarray(em_df.iloc[:, 1:])
    # embedding = np.row_stack((embedding, embedding.mean(axis=0)))
    # return word_dict, embedding
    save_embedding(dir, name, word_dict, embedding)

    return word_dict, embedding


def save_embedding(path, name, word_dict, embedding, resave=False):
    # glove.6b.dict.pkl
    # glove.840b.dict.pkl
    dict_filename = os.path.join(path, name + '.dict.pkl')
    if resave or not os.path.exists(dict_filename):
        with open(dict_filename, 'wb') as f:
            pickle.dump(word_dict, f)

    # glove.6b.embd.npy
    # glove.840b.embd.npy
    embd_filename = os.path.join(path, name + '.embd.npy')
    if resave or not os.path.exists(embd_filename):
        with open(embd_filename, 'wb') as f:
            np.save(f, embedding)


def load_embedding(path, name):
    dict_filename = os.path.join(path, name + '.dict.pkl')
    embd_filename = os.path.join(path, name + '.embd.npy')
    if not os.path.exists(dict_filename) or not os.path.exists(embd_filename):
        pprint('Can not found pickled embedding file.')
        pprint('Need to extract and pickle embedding file.')
        return None, None
    with open(dict_filename, 'rb') as d_f, open(embd_filename, 'rb') as n_f:
        pprint('Loading pickled embedding.')
        word_dict = pickle.load(d_f)
        embedding = np.load(n_f)
    return word_dict, embedding

if __name__ == '__main__':
    from config import DATA_DIR
    import os
    # 2196018
    # 400001
    file_path = os.path.join(DATA_DIR, 'Glove/glove.840B.300d.zip')
    di_1, embd_1 = build_voc(file_path,  2196018)

    # file_path = os.path.join(DATA_DIR, 'Glove/glove.6B.300d.zip')
    # di_1, embd_1 = build_voc(file_path,  400001)
    print(len(di_1))
    print(embd_1)
    pprint(embd_1.shape)


    # di_2, embd_2 = voc_builder(os.path.join(DATA_DIR, 'Glove/glove.6B.300d.txt'))

    # print(embd_1[-1, :] - embd_2[-1, :])

# 15291 11862 16640 64611 122665, 140648 138700

