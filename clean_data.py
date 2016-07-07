import os
import pandas as pd


path = os.getcwd()
dic_filepath = os.path.join(path, 'datasets/SST/dictionary.txt')
label_filepath = os.path.join(path, 'datasets/SST/sentiment_labels.txt')
sentence_filepath = os.path.join(path, 'datasets/SST/datasetSentences-cleaned.txt')

dict_phase = dict()
dict_sentiment = dict()

with open(dic_filepath, 'r', encoding='utf-8') as f:
    i = 0
    f.readline()
    while True:
        line = f.readline()
        i += 1
        if line is None or line == '':
            break
        phase = line.split('|')[0].strip()
        id = int(line.split('|')[1])
        dict_phase[id] = phase

rev_dict_phase = dict((v, k) for k, v in dict_phase.items())

with open(label_filepath, 'r', encoding='utf-8') as f:
    i = 0
    f.readline()
    while True:
        line = f.readline()
        i += 1
        if line is None or line == '':
            break

        id = int(line.split('|')[0])
        if id in dict_phase:
            sentiment = float(line.split('|')[1])
            dict_sentiment[id] = sentiment
        else:
            print('unk id!')

with open(sentence_filepath, 'r', encoding='iso-8859-1') as f:
    i = 0
    f.readline()
    count = 0
    while True:
        line = f.readline()
        i += 1
        if line is None or line == '':
            break

        id = int(line.split('\t')[0])
        sentence = line.split('\t')[1].strip()

        if sentence not in rev_dict_phase:
            print('Unknown sentence:')
            print(sentence)
            count += 1

print(count)
# print(dict_phase)
# print(len(dict_sentiment))
# sentences_df = pd.read_csv(sentence_filepath, sep='\t', quoting=3, encoding='latin')
# print(sentences_df)