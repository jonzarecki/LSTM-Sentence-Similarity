import pickle
import numpy as np
from sklearn.svm import SVC

from lstm import lstm
from util_files.Constants import data_folder, use_noise, models_folder
from util_files.data_utils import get_discrete_accuracy



def prepare_entailment_data(data):
    return map(lambda entry: [entry[0], entry[1], entry[3]], data)

def prepare_svm_data(mydata, lst):
    # type: (list, lstm) -> tuple
    num = len(mydata)
    features = []
    ys = []
    use_noise.set_value(0.)

    for idx in range(0, num):  # I don't use batches to make calculating each feature vector easier
        [sent1, sent2, y] = mydata[idx]
        emb1 = lst.get_sentence_embedding(sent1)
        emb2 = lst.get_sentence_embedding(sent2)
        feat_vect = np.append(np.fabs(emb1-emb2), [emb1*emb2])  # as described in the orig paper
        ys.append(y)
        features.append(feat_vect)

    features = np.array(features)
    ys = np.array(ys)
    return features, ys

model_name = "bestsem.p"
print model_name
lst=lstm(model_path=models_folder + model_name, load=True)

train = pickle.load(open(data_folder + "semtrain.p", 'rb'))
train = prepare_entailment_data(train)
test = pickle.load(open(data_folder + "semtest.p", 'rb'))
test = prepare_entailment_data(test)

x_train, y_train = prepare_svm_data(train, lst)
x_test, y_test = prepare_svm_data(test, lst)

clf = SVC(C=100, gamma=3.1, kernel='rbf')
clf.fit(x_train, y_train)

print "Training accuracy:", get_discrete_accuracy(clf, x_train, y_train)
print "Test accuracy:", get_discrete_accuracy(clf, x_test, y_test)
