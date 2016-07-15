import os

"""
Some important path
"""

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'datasets')

GLOVE_840B_PATH = os.path.join(DATA_DIR, 'Glove/glove.840B.300d.zip')
GLOVE_6B_PATH = os.path.join(DATA_DIR, 'Glove/glove.6B.300d.zip')

SST_CLEANED_840B_PATH = os.path.join(DATA_DIR, 'Diy/sst/840b')
SST_CLEANED_6B_PATH = os.path.join(DATA_DIR, 'Diy/sst/6b')

SNLI_CLEANED_840B_PATH = os.path.join(DATA_DIR, 'Diy/snli/840b')


def info_(**kwargs):
    return kwargs

if __name__ == '__main__':
    import re
    [print('%-25s:%s' % (k, v)) for k, v in locals().copy().items() if re.match(r"^[A-Z].*[A-Z]$", k)]