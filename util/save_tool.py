from config import ROOT_DIR
import os
from datetime import datetime
import time
import tensorflow as tf
import json
import logging
import inspect


class ResultSaver:
    def __init__(self, model_name, model=None, savePara=False, sess=None):
        OUT_ROOT = os.path.join(ROOT_DIR, 'runs')
        timestamp = '{0:(%Y-%m-%d-%H:%M:%S)}'.format(datetime.now())
        self.model = model
        self.saver_root = os.path.abspath(os.path.join(OUT_ROOT, '-'.join(['result', model_name, timestamp])))
        self.checkpoint_dir = os.path.abspath(os.path.join(self.saver_root, "checkpoints"))
        self.meta_filename = os.path.abspath(os.path.join(self.saver_root, "meta.json"))
        self.model_src_filename = os.path.abspath(os.path.join(self.saver_root, "model_src.py"))
        self.log_filename = os.path.abspath(os.path.join(self.saver_root, "log.txt"))
        self.log_file = None
        self.epoch_filename = os.path.abspath(os.path.join(self.saver_root, 'epoch_info.txt'))
        self.savePara = savePara
        self.sess = sess
        self.tf_saver = None

    def setup(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # self.log_file = open(self.log_filename, 'w', encoding='utf-8')

        with open(self.meta_filename, 'w', encoding='utf-8') as meta_f:
            json.dump(obj=self.model.model_info, sort_keys=True, indent=4, fp=meta_f)
            print("Saved model meta-info to {}".format(self.log_filename))
        with open(self.model_src_filename, 'w', encoding='utf-8') as src_f:
            model_src = inspect.getsource(type(self.model))
            src_f.write(model_src)
            src_f.flush()

        if self.savePara:
            self.tf_saver = tf.train.Saver(tf.all_variables())

    def logging(self, msg, with_time=True):
        if with_time:
            logging.basicConfig(filename=self.log_filename, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
            logging.info(msg)
        else:
            logging.basicConfig(filename=self.log_filename, level=logging.INFO, format='%(message)s')
            logging.info(msg)
        print("Logging stats to {}".format(self.log_filename))

    def logging_epoch(self, msg):
        logging.basicConfig(filename=self.log_filename, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
        logging.info(msg)
        # self.log_file.write(msg + '\n')
        # if with_time:
        #     timestamp = '{0:(%Y-%m-%d-%H:%M:%S)}'.format(datetime.now())
        #     self.log_file.write(' '.join(['Time:', timestamp, '\n']))
        #     self.log_file.flush()
        # print("Saved prediction stats to {}".format(self.log_file))

    def save_params(self, info=None, with_time=True):
        timestamp = str(int(time.time())) + '.ckpt'
        path = self.tf_saver.save(self.model.sess, os.path.join(self.checkpoint_dir, timestamp))
        pathinfo = ' '.join(['Path:', path])
        self.logging(pathinfo, with_time=False)
        self.logging(info, with_time=with_time)
        self.log_file.flush()
        print("Saved model checkpoint to {}".format(path))

    def close(self):
        self.log_file.close()

if __name__ == '__main__':
    from models.snli.baselineModels_snli import SnliBasicLSTM
    model = SnliBasicLSTM()
    recorder = ResultSaver('Test', model=model)
    recorder.setup()
    recorder.logging('testing')
    recorder.close()