import pickle

import os
import theano
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

from util_files.general_utils import numpy_floatX


prefix = 'lstm'
noise_std = 0.
use_noise = theano.shared(numpy_floatX(0.))
flg = 1
cachedStopWords = stopwords.words("english")
training = True  # Loads best saved model if False
Syn_aug = True  # If true, performs better on Test dataset but longer training time
model = None
options=locals().copy()


data_folder = "data/"
embeddings_folder = os.path.join(data_folder, "word_embeddings/")

def initialize_w2v():
    global model
    if model is None:
        print "Loading word2vec"
        model = KeyedVectors.load_word2vec_format(embeddings_folder + "GoogleNews-vectors-negative300.bin.gz", binary=True)
        print "Finished loading word2vec"
initialize_w2v()

d2 = pickle.load(open(data_folder + "synsem.p", 'rb'))
dtr = pickle.load(open(data_folder + "dwords.p", 'rb'))
# d2=dtr
# model=pickle.load(open("Semevalembed.p","rb"))


# In[7]:

## import pickle
# from random import shuffle


# In[9]:
