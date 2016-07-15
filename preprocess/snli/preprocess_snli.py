import json
import os
import re
import sys


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """

    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r";", " ; ", string)


    # Newly added
    # string = re.sub(r"\'[^svtrdl]", " \' ", string)
    string = re.sub(r'"', ' " ', string)
    # string = re.sub(r"\'", " \' ", string)

    return string.strip().lower()


def read_raw_file(file):
    data = list()
    with open(file, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if line == '':
                break
            item = json.loads(line)
            data.append(item)

    return data


def sentence2id(data, word_dict):
    result = []
    unk_count = 0
    for item in data:
        new_item = {}
        sentence_words_1 = clean_str(item['sentence1']).split()
        sentence_words_2 = clean_str(item['sentence2']).split()
        for i, word in enumerate(sentence_words_1):
            if word not in word_dict:
                unk_count += 1
                sentence_words_1[i] = '<unk>'
        sentence_ids_1 = [word_dict[word] for word in sentence_words_1]

        for i, word in enumerate(sentence_words_2):
            if word not in word_dict:
                unk_count += 1
                sentence_words_2[i] = '<unk>'
        sentence_ids_2 = [word_dict[word] for word in sentence_words_2]

        new_item['sentence1'] = sentence_ids_1
        new_item['length1'] = len(sentence_ids_1)
        new_item['sentence2'] = sentence_ids_2
        new_item['length2'] = len(sentence_ids_2)
        new_item['label'] = item['gold_label']
        # print(new_item)

        result.append(new_item)
    return result, unk_count


def write_clean_data(data, file):
    label_d = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

    with open(file, 'w', encoding='utf-8') as f:
        f.write('sentence1|length1|sentence2|length2|label\n')
        count_not_gold_label = 0

        for item in data:
            sentence_id_1 = item['sentence1']
            sentence_id_2 = item['sentence2']
            label = item['label']

            # 2% of the label
            if label not in label_d:
                count_not_gold_label += 1
                continue

            f.write(' '.join([str(_id) for _id in sentence_id_1]))
            f.write('|')
            f.write(str(item['length1']))
            f.write('|')
            f.write(' '.join([str(_id) for _id in sentence_id_2]))
            f.write('|')
            f.write(str(item['length2']))
            f.write('|')
            f.write(str(label_d[label]))
            f.write('\n')

        print('Number of item with no glod-label:', count_not_gold_label)

if __name__ == '__main__':
    from util.dict_builder import fine_selected_embedding
    from util.vocab_stat import get_dict
    from config import DATA_DIR

    train_raw_file = os.path.join(DATA_DIR, 'SNLI/snli_1.0_train.jsonl')
    dev_raw_file = os.path.join(DATA_DIR, 'SNLI/snli_1.0_dev.jsonl')
    test_raw_file = os.path.join(DATA_DIR, 'SNLI/snli_1.0_test.jsonl')

    test_data = read_raw_file(test_raw_file)
    dev_data = read_raw_file(dev_raw_file)
    train_data = read_raw_file(train_raw_file)

    fine_dict = get_dict()
    word2id, word_embedding = fine_selected_embedding(os.path.join(DATA_DIR, 'Glove/glove.840B.300d.zip'),
                                                      fine_dict, pre_vocab_size=2196018)

    embd_test_data, count_t = sentence2id(test_data, word2id)
    embd_dev_data, count_d = sentence2id(dev_data, word2id)
    embd_train_data, count_tr = sentence2id(train_data, word2id)
    print('Number of unknown words:')
    print(count_tr, 'in train')
    print(count_t, 'in test')
    print(count_d, 'in dev')

    write_clean_data(embd_test_data, os.path.join(DATA_DIR, 'Diy/snli/840b/test_data.txt'))
    write_clean_data(embd_dev_data, os.path.join(DATA_DIR, 'Diy/snli/840b/dev_data.txt'))
    write_clean_data(embd_train_data, os.path.join(DATA_DIR, 'Diy/snli/840b/train_data.txt'))