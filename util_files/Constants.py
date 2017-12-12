import pickle
import os
import theano
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

from util_files.general_utils import numpy_floatX


data_folder = "data/"
embeddings_folder = os.path.join(data_folder, "word_embeddings/")
models_folder = os.path.join(data_folder, "models/")
prefix = 'lstm'
noise_std = 0.
use_noise = theano.shared(numpy_floatX(0.))
flg = 1
cachedStopWords = stopwords.words("english")
model = None
word_prob = None
total_counts = None

def initialize_word_prob():
    global word_prob
    if word_prob is None:
        word_prob = dict()
        for line in open(embeddings_folder + "count_1w.txt"):
            word, count = line.split()
            word_prob[word] = float(count)
        global total_counts
        total_counts = sum(word_prob.itervalues())
        for word, count in word_prob.iteritems():  # not probability just yet
            word_prob[word] = word_prob[word] / total_counts
initialize_word_prob()


def initialize_w2v():
    global model
    if model is None:
        print "Loading word2vec"
        model = KeyedVectors.load_word2vec_format(embeddings_folder + "GoogleNews-vectors-negative300.bin.gz", binary=True)
        print "Finished loading word2vec"
initialize_w2v()

d2 = pickle.load(open(data_folder + "synsem.p", 'rb'))
dtr = pickle.load(open(data_folder + "dwords.p", 'rb'))
