import os

"""
Some important path
"""

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'datasets')

GLOVE_840B_PATH = os.path.join(DATA_DIR, 'Glove/glove.840B.300d.zip')
GLOVE_6B_PATH = os.path.join(DATA_DIR, 'Glove/glove.6B.300d.zip')

GLOVE_6B_PATH_VOCAB_SIZE = 400001
GLOVE_840B_PATH_VOCAB_SIZE = 2196018
GLOVE_FS_PATH_VOCAB_SIZE = 40890

SST_CLEANED_840B_PATH = os.path.join(DATA_DIR, 'Diy/sst/840b')
SST_CLEANED_6B_PATH = os.path.join(DATA_DIR, 'Diy/sst/6b')

SNLI_CLEANED_840B_PATH = os.path.join(DATA_DIR, 'Diy/snli/840b')

SNLI_CLEANED_840B_TRAIN_SET_FILE = os.path.join(SNLI_CLEANED_840B_PATH, 'train_data.txt')
SNLI_CLEANED_840B_DEV_SET_FILE = os.path.join(SNLI_CLEANED_840B_PATH, 'dev_data.txt')
SNLI_CLEANED_840B_TEST_SET_FILE = os.path.join(SNLI_CLEANED_840B_PATH, 'test_data.txt')

SNLI_840B_H5_TRAIN_FILE = os.path.join(DATA_DIR, 'Diy/snli/train_data.h5')
SNLI_840B_H5_TEST_FILE = os.path.join(DATA_DIR, 'Diy/snli/test_data.h5')
SNLI_840B_H5_DEV_FILE = os.path.join(DATA_DIR, 'Diy/snli/dev_data.h5')

SNLI_SKIPTH_TEST_FILE_ON_MAC = os.path.join(DATA_DIR, 'Diy/snli_skipth_data/snli_test_data.h5')
SNLI_SKIPTH_DEV_FILE_ON_MAC = os.path.join(DATA_DIR, 'Diy/snli_skipth_data/snli_dev_data.h5')
SNLI_SKIPTH_TRAIN_FILE_ON_MAC = os.path.join(DATA_DIR, 'Diy/snli_skipth_data/snli_train_data.h5')

SLURM_DATA_ROOT = '/share/data/ripl'

SNLI_SKIPTH_TRAIN_FILE_ON_SLURM = os.path.join(SLURM_DATA_ROOT, 'snli_skipth_data/snli_train_data.h5')
SNLI_SKIPTH_DEV_FILE_ON_SLURM = os.path.join(SLURM_DATA_ROOT, 'snli_skipth_data/snli_dev_data.h5')
SNLI_SKIPTH_TEST_FILE_ON_SLURM = os.path.join(SLURM_DATA_ROOT, 'snli_skipth_data/snli_test_data.h5')


def info_(**kwargs):
    return kwargs

if __name__ == '__main__':
    import re
    [print('%-25s:%s' % (k, v)) for k, v in locals().copy().items() if re.match(r"^[A-Z].*[A-Z]$", k)]