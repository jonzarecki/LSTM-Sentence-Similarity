from random import shuffle

import pickle
import numpy as np
from sklearn.svm import SVC

from lstm import lstm
from util_files.Constants import data_folder, use_noise, models_folder
from util_files.data_utils import prepare_single_sent_data, get_discrete_accuracy

model_name = "bestsem.p"
print model_name
lst=lstm(model_path=models_folder + model_name, load=True)
train = pickle.load(open(data_folder + "kaggle.p", 'rb'))


def prepare_svm_train_data(mydata):
    num = len(mydata)
    px = []
    yx = []
    use_noise.set_value(0.)
    for batch_st_idx in range(0, num, 64):  # use batch to make calculations faster
        batch_data = []
        batch_end = batch_st_idx + 64
        if batch_end > num:
            batch_end = num
        for j in range(batch_st_idx, batch_end):
            batch_data.append(mydata[j])
        x1, mas1, y2 = prepare_single_sent_data(batch_data)
        sents = map(lambda tpl: tpl[0], batch_data)
        pred = lst.get_sentence_embedding_bulk(sents)

        for z in range(0, len(batch_data)):
            yx.append(y2[z])
            px.append(pred[z])
    px = np.array(px)
    yx = np.array(yx)
    return px, yx


shuffle(train)
xdat, ydat = prepare_svm_train_data(train)
train_lim = int(0.7 * len(xdat))

x_train = xdat[0:train_lim]
y_train = ydat[0:train_lim]
x_cross_val = xdat[train_lim:]
y_cross_val = ydat[train_lim:]

clf = SVC(C=100, gamma=3.1, kernel='rbf')
clf.fit(x_train, y_train)

print "Training accuracy:", get_discrete_accuracy(clf, x_train, y_train)
print "Cross validation accuracy:", get_discrete_accuracy(clf, x_cross_val, y_cross_val)
