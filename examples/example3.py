import sys

from lstm import *
from util_files.Constants import data_folder, models_folder
from util_files.data_utils import expand


training = True  # Set to false to load weights
Syn_aug = False  # it False faster but does slightly worse on Test dataset
save_model = False

sls = lstm(models_folder + "new.p", load=False, training=True)

train = pickle.load(open(data_folder + "stsallrmf.p", "rb"))  # [:-8]
if training:
    print "Pre-training"
    sls.train_lstm(train, 66)  # 66 epochs of pre-training
    print "Pre-training done"
    train = pickle.load(open(data_folder + "semtrain.p", 'rb'))
    if Syn_aug:
        train = expand(train, ignore_flag=True)
        sls.train_lstm(train, 375)
    else:
        sls.train_lstm(train, 330)

if save_model:
    sys.setrecursionlimit(5000)  # avoid limit-exceeded when pickling
    pickle.dump(sls, open(models_folder + "datanewp.p", "wb"))

test = pickle.load(open(data_folder + "semtest.p", 'rb'))
print sls.chkterr2(test)
# Example
sa = "A truly wise man"
sb = "He is smart"
print sls.predict_similarity(sa, sb) * 4.0 + 1.0
