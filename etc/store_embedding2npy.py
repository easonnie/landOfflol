
if __name__ == '__main__':
    import numpy as np
    from config import DATA_DIR
    import pickle
    import os
    from os.path import basename
    from pprint import pprint

    # 2196018
    # 400001

    # file_path = os.path.join(DATA_DIR, 'Glove/glove.840B.300d.zip')
    # di_840b, embd_840b = build_voc(file_path,  2196018)
    save_path = os.path.join(DATA_DIR, 'Diy/glove')
    # test
    test_di = {'a': 1}
    test_nd = np.asarray([np.random.randint(0, 5, 10) for i in range(3)])

    # with open(save_path + '/test.pkl', 'wb') as d_f, open(save_path + '/test.npy', 'wb') as n_f:
    #     pickle.dump(test_di, d_f)
    #     np.save(n_f, test_nd)
    #
    # with open(save_path + '/test.pkl', 'rb') as d_f, open(save_path + '/test.npy', 'rb') as n_f:
    #     in_di = pickle.load(d_f)
    #     in_nd = np.load(n_f)
    #     pprint(in_di)
    #     pprint(in_nd)
    pprint(basename('/Users/Eason/RA/landOfflol/datasets/Glove/glove.6B.300d.txt.zip'))