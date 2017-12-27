import pickle
import random
import os
import datetime
import numpy

import negative_sampling
import util_files.Constants as cn
from lstm import lstm
from negative_sampling import extend_negative_samples
from util_files import file_util
from util_files.Constants import data_folder
from util_files.positive_expansion import expand_positive_examples
from util_files.run_experiment_util import experiment_on_data_and_save_results
from util_files.tee import Tee

random.seed(1554)
numpy.random.seed(42)


def experiment(theme):
    Syn_aug = True  # it False faster but does slightly worse on Test dataset
    save_model = True

    model_name = "negative5000_negscore"+str(negative_sampling.negative_score)+\
                 "genamount"+str(negative_sampling.new_examples_amout)+"_model"
    epoch_num_pre_training = 0
    epoch_num_training = 300
    print model_name
    print
    print "epoch num pre-training: " + str(epoch_num_pre_training)
    print "epoch num training: " + str(epoch_num_training)

    sls = lstm(training=True)
    test = pickle.load(open(data_folder + "semtest.p", 'rb'))
    train = pickle.load(open(data_folder + "stsallrmf.p", "rb"))

    print "Loading pre-training model"
    sls.train_lstm(train, epoch_num_pre_training)
    print "Pre-training done"
    train = pickle.load(open(data_folder + "semtrain.p", 'rb'))
    if Syn_aug:
        print "Train with negative sampling"
        train_enriched = extend_negative_samples(train)
        # print "Train with positive sampling"
        # train_enriched = expand_positive_examples(train_enriched, ignore_flag=True)
        sls.train_lstm(train_enriched, epoch_num_training, eval_data=test, disp_freq=25)
    else:
        print "Train normaly"
        sls.train_lstm(train, epoch_num_training, eval_data=test)

    print sls.check_error(test)

    if save_model:
        sls.save_to_pickle(cn.tmp_expr_foldpath + "/" + model_name + ".p")

    # Example
    sa = "A truly wise man"
    sb = "He is smart"
    print sls.predict_similarity(sa, sb) * 4.0 + 1.0

    return theme + model_name

if __name__ == '__main__':
    purpose = "range_neg_amount_random_prob/"
    expr_id = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')  # used in all files to identify this experiment

    tmp_expr_folder_prefix = "/tmp/SiameseLSTM/" + purpose
    file_util.makedirs(tmp_expr_folder_prefix, exists_ok=True)
    tmp_expr_foldpath = file_util.create_temp_folder(prefix=tmp_expr_folder_prefix)

    with Tee(os.path.join(tmp_expr_foldpath, 'output_log' + expr_id + '.txt')):
        # experiment_on_data_and_save_results(experiment, 0)
        purpose = "range_neg_amount_random_prob/"
        print purpose
        # for neg_score in numpy.arange(1.0, 5.5, 0.5):
        #     negative_sampling.negative_score = neg_score
        #     experiment_on_data_and_save_results(lambda: experiment(purpose), 0)

        for neg_amount in range(1000, 16000, 1000):
            negative_sampling.new_examples_amout = neg_amount
            experiment_on_data_and_save_results(lambda: experiment(purpose), 0)

        for neg_amount in range(20000, 65000, 5000):
            negative_sampling.new_examples_amout = neg_amount
            experiment_on_data_and_save_results(lambda: experiment(purpose), 0)

    print ""