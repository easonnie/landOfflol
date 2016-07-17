from preprocess.snli.batch_generator_snli import BatchGenerator
import config
from util.vocab_stat import get_dict
from util.dict_builder import fine_selected_embedding
import numpy as np


def setup():
    word_dict = get_dict()
    word2id, embedding = fine_selected_embedding(config.GLOVE_840B_PATH, word_dict,
                                                 pre_vocab_size=config.GLOVE_840B_PATH_VOCAB_SIZE)
    return word2id, embedding

def basic_test():
    print('Basic test')
    train_batchGenerator = BatchGenerator(config.SNLI_CLEANED_840B_TRAIN_SET_FILE, maxlength=20)
    one_batch = train_batchGenerator.next_batch(3)
    premise, p_len, hypothesis, h_len, label = one_batch
    print('Type:', type(one_batch))
    print('Shape(premise/p_length/hypothesis/h_length/label):',
           premise.shape, p_len.shape, hypothesis.shape, h_len.shape, label.shape)
    print('Premise:\n', premise)
    print('P_length:', p_len)
    print('Hypothesis:\n', hypothesis)
    print('H_length:', h_len)
    print('Label:', label)
    train_batchGenerator.close()

    print('Reopen test')
    one_batch = train_batchGenerator.next_batch(3)
    premise, p_len, hypothesis, h_len, label = one_batch
    print('Type:', type(one_batch))
    print('Shape(premise/p_length/hypothesis/h_length/label):',
           premise.shape, p_len.shape, hypothesis.shape, h_len.shape, label.shape)
    print('Premise:\n', premise)
    print('P_length:', p_len)
    print('Hypothesis:\n', hypothesis)
    print('H_length:', h_len)
    print('Label:', label)
    train_batchGenerator.close()


def embedding_test(word2id):
    id2word = {v: k for k, v in word2id.items()}

    dev_batchGenerator = BatchGenerator(config.SNLI_CLEANED_840B_DEV_SET_FILE, maxlength=100)
    test_batchGenerator = BatchGenerator(config.SNLI_CLEANED_840B_TEST_SET_FILE, maxlength=100)
    train_batchGenerator = BatchGenerator(config.SNLI_CLEANED_840B_TRAIN_SET_FILE, maxlength=100)
    dev_result = dev_batchGenerator.next_batch(-1)
    test_result = test_batchGenerator.next_batch(-1)
    train_result = train_batchGenerator.next_batch(100)

    s1, l1, s2, l2, b = train_result
    sample = np.random.randint(0, s1.shape[0], 100)
    for row in s1[sample, :]:
        line = ' '.join([id2word[i] for i in row])
        print(line)
    for row in s2[sample, :]:
        line = ' '.join([id2word[i] for i in row])
        print(line)



def dev_test_setNum_test():
    dev_batchGenerator = BatchGenerator(config.SNLI_CLEANED_840B_DEV_SET_FILE, maxlength=80)
    test_batchGenerator = BatchGenerator(config.SNLI_CLEANED_840B_TEST_SET_FILE, maxlength=80)
    train_batchGenerator = BatchGenerator(config.SNLI_CLEANED_840B_TRAIN_SET_FILE, maxlength=100)

    dev_result = dev_batchGenerator.next_batch(-1)
    test_result = test_batchGenerator.next_batch(-1)
    train_result = train_batchGenerator.next_batch(-1, save=False)

    s1, l1, s2, l2, b = dev_result
    print(s1.shape)
    # print(s2.shape)
    s1, l1, s2, l2, b = test_result
    print(s1.shape)
    s1, l1, s2, l2, b = train_result
    print(train_result.shape)

if __name__ == '__main__':
    """
    basic test
    """
    word2id, embedding = setup()
    # basic_test()
    # dev_test_setNum_test()
    embedding_test(word2id)


