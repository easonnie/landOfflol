import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'datasets')

if __name__ == '__main__':
    """
    Setup evn
    """
    os.system("export PYTHONPATH=%s" % ROOT_DIR)

    print('ROOT_DIR:', ROOT_DIR)
    print('DATA_DIR:', DATA_DIR)