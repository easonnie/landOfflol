from datetime import datetime
from tests.snli.skipth_test_wrapper import wrapper
from models.snli.baselineModels_snli import BasicSkipThought

import argparse

if __name__ == '__main__':
    arg_parse = argparse.ArgumentParser(description='Test for some variant of LSTM model.')
    arg_parse.add_argument('-m', '--message', metavar='message', dest='message', action='store',
                           help='Specify a message for this test')

    args = arg_parse.parse_args()

    timestamp = '{0:(%Y-%m-%d-%H:%M:%S)}'.format(datetime.now())

    model = BasicSkipThought(sentence_d=4800, softmax_keeprate=0.70, learning_rate=0.001, Message=args.message)
    model.setup()
    wrapper(model_name='skipth-cat', model=model, batch_size=256, benchmark=0.70)