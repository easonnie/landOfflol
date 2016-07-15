from config import DATA_DIR
import os
import pickle
import json
from preprocess.snli.preprocess_snli import clean_str
import re


def get_dict(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, 'Diy/etc/dict.pkl')

    if os.path.exists(path):
        with open(path, 'rb') as f:
            dict = pickle.load(f)
    else:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        dict = set()
        with open(path, 'wb') as f:
            pickle.dump(dict, f)
    return dict


def save_dict(dict, path=None):
    if path is None:
        path = os.path.join(DATA_DIR, 'Diy/etc/dict.pkl')
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(dict, f)


def absorb_snli_words(path, dict):
    with open(path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if line == '':
                break
            item = json.loads(line)
            sentence_words_1 = clean_str(item['sentence1']).split()
            sentence_words_2 = clean_str(item['sentence2']).split()
            for word in sentence_words_1:
                dict.add(word)
            for word in sentence_words_2:
                dict.add(word)


def count_unseen(path, dict):
    with open(path, 'r', encoding='utf-8') as f:
        i = 0
        while True:
            line = f.readline()
            if line == '':
                break
            item = json.loads(line)
            sentence_words_1 = clean_str(item['sentence1']).split()
            sentence_words_2 = clean_str(item['sentence2']).split()
            for word in sentence_words_1:
                if word not in dict:
                    # print(word)
                    i += 1
            for word in sentence_words_2:
                if word not in dict:
                    # print(word)
                    i += 1
        print('Unseen words in', path, ':', i)


def count_unseen_sst(path, dict):
    with open(sst_data_path, 'r', encoding='iso-8859-1') as f:
        f.readline()
        i = 0
        while True:
            line = f.readline()
            if line == '':
                break
            line = re.sub(r'"', ' " ', line)
            for word in line.lower().split()[1:]:
                if word not in dict:
                    i += 1
        print('Unseen words in', path, ':', i)

if __name__ == '__main__':

    whole_dict = get_dict()

    if len(whole_dict) == 0:
        sst_data_path = os.path.join(DATA_DIR, 'SST/datasetSentences-cleaned.txt')
        with open(sst_data_path, 'r', encoding='iso-8859-1') as f:
            f.readline()
            while True:
                line = f.readline()
                if line == '':
                    break
                line = re.sub(r'"', ' " ', line)
                for word in line.lower().split()[1:]:
                    whole_dict.add(word)
        absorb_snli_words(os.path.join(DATA_DIR, 'SNLI/snli_1.0_dev.jsonl'), whole_dict)
        absorb_snli_words(os.path.join(DATA_DIR, 'SNLI/snli_1.0_test.jsonl'), whole_dict)
        absorb_snli_words(os.path.join(DATA_DIR, 'SNLI/snli_1.0_train.jsonl'), whole_dict)

    from util.dict_builder import build_voc
    file_path = os.path.join(DATA_DIR, 'Glove/glove.840B.300d.zip')
    dict, embd = build_voc(file_path,  2196018)

    count_unseen(os.path.join(DATA_DIR, 'SNLI/snli_1.0_dev.jsonl'), dict)
    count_unseen(os.path.join(DATA_DIR, 'SNLI/snli_1.0_test.jsonl'), dict)
    count_unseen(os.path.join(DATA_DIR, 'SNLI/snli_1.0_train.jsonl'), dict)

    sst_data_path = os.path.join(DATA_DIR, 'SST/datasetSentences-cleaned.txt')
    count_unseen_sst(sst_data_path, dict)

    save_dict(whole_dict)

    """
    Glove 6B:
    Unseen words in /Users/Eason/RA/landOfflol/datasets/SNLI/snli_1.0_dev.jsonl : 600
    Unseen words in /Users/Eason/RA/landOfflol/datasets/SNLI/snli_1.0_test.jsonl : 640
    Unseen words in /Users/Eason/RA/landOfflol/datasets/SNLI/snli_1.0_train.jsonl : 30255
    Unseen words in /Users/Eason/RA/landOfflol/datasets/SST/datasetSentences-cleaned.txt : 2048

    Glove 840B:
    Unseen words in /Users/Eason/RA/landOfflol/datasets/SNLI/snli_1.0_dev.jsonl : 263
    Unseen words in /Users/Eason/RA/landOfflol/datasets/SNLI/snli_1.0_test.jsonl : 259
    Unseen words in /Users/Eason/RA/landOfflol/datasets/SNLI/snli_1.0_train.jsonl : 13664
    Unseen words in /Users/Eason/RA/landOfflol/datasets/SST/datasetSentences-cleaned.txt : 1767
    """