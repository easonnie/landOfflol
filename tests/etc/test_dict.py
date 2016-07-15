from util.vocab_stat import get_dict
from util.dict_builder import build_voc, fine_selected_embedding
from config import DATA_DIR
import os

if __name__ == '__main__':
    # print(len(get_dict()))
    # if '\\' in get_dict():
    #     print('hehe')

    dict_ = get_dict()
    file_path = os.path.join(DATA_DIR, 'Glove/glove.840B.300d.zip')
    di_1, embd_1 = build_voc(file_path, 2196018)

    print('Dict:', len(dict_))
    print('6B glove dict:', len(di_1), embd_1.shape)

    i = 0
    for word in di_1:
        if word in dict_:
            i += 1
    print('inter', i)

    di, embd = fine_selected_embedding(file_path, dict_, pre_vocab_size=2196018)
    print('Fine', len(di), embd.shape)
    # 'distaste': 20964
    print(di)
    print(embd[20964, :])


    # file_path = os.path.join(DATA_DIR, 'Glove/glove.840B.300d.zip')
    # di_1, embd_1 = build_voc(file_path,  2196018)