import numpy as np
import pandas as pd


path = '/Users/Eason/RA/landOfflol/'
em = pd.read_csv(path + 'datasets/Glove/glove.6B.300d.txt', sep=' ', header=None, quoting=3, encoding='utf-8')

def voc_builder(vocab_path):
    em = pd.read_csv(vocab_path)

