import numpy
import numpy as np

from util_files.Constants import dtr, model, word_prob, total_counts


def prepare_sent_pairs_data(data):
    xa1 = []
    xb1 = []
    y2 = []
    for i in range(0, len(data)):
        xa1.append(data[i][0])
        xb1.append(data[i][1])
        y2.append(data[i][2])
    lengths = []
    for i in xa1:
        lengths.append(len(i.split()))
    for i in xb1:
        lengths.append(len(i.split()))
    maxlen = numpy.max(lengths)
    x1, mas1 = getmtr(xa1, maxlen)
    x2, mas2 = getmtr(xb1, maxlen)

    y2 = np.array(y2, dtype=np.float32)

    assert len(data) == len(x1), "_prepare_embeddings assertion broken"
    return x1, mas1, x2, mas2, y2


def prepare_single_sent_data(data):
    xa1 = []
    y2 = []
    for i in range(0,len(data)):
        xa1.append(data[i][0])
        y2.append(data[i][1])
    lengths=[]
    for i in xa1:
        lengths.append(len(i.split()))
    maxlen = numpy.max(lengths)
    x1, mas1 = getmtr(xa1, maxlen)

    return x1, mas1, y2


def getmtr(xa, maxlen):
    n_samples = len(xa)
    ls = []
    x_mask = numpy.zeros((maxlen, n_samples)).astype(np.float32)
    for i in range(0, len(xa)):
        q = xa[i].split()
        for j in range(0, len(q)):
            x_mask[j][i] = 1.0
        while len(q) < maxlen:
            q.append(',')
        ls.append(q)
    xa = np.array(ls)
    return xa, x_mask

# def pkl_to_csv(filename):
#     data = pickle.load(open(data_folder + filename, "rb"))
#     df = pd.DataFrame(data, columns=['sent1', 'sent2', 'sim_score'])
#     df.to_csv(data_folder + filename + ".csv")


def embed_sentence(sent_arr):
    """ embed sent_arr (which is a numpy array with the words array(['A', 'truly', 'wise', 'man'], dtype='|S5') """
    dmtr = numpy.zeros((sent_arr.shape[0], 300), dtype=np.float32)
    word_idx = 0
    while word_idx < len(sent_arr):
        if sent_arr[word_idx] == ',':
            word_idx += 1
            continue
        if sent_arr[word_idx] in dtr:
            dmtr[word_idx] = model[dtr[sent_arr[word_idx]]]
            word_idx += 1
        else:
            dmtr[word_idx] = model[sent_arr[word_idx]]
            word_idx += 1
    return dmtr


def prepare_sent_pair_word_embeddings(x1, x2):
    assert len(x1) == len(x2), "new function not equal to old one"
    return prepare_sent_word_embedding(x1), prepare_sent_word_embedding(x2)


def prepare_sent_word_embedding(sent_list):
    ls = []
    for sent_arr in sent_list:
        ls.append(embed_sentence(sent_arr))
    trconv = np.dstack(ls)
    sents_word_emb = np.swapaxes(trconv, 1, 2)
    return sents_word_emb


def sentence_unigram_probability(sent):
    """ pretty weak language model but should be enough"""
    prob = 1
    for word in sent.split():
        if word in word_prob:
            prob *= word_prob[word]
        else:
            prob *= 1.0 / total_counts
    return prob


def get_discrete_accuracy(classif, x_test, y_test):
    pred = classif.predict(x_test)
    score = np.sum(y_test == pred) / float(len(y_test)) * 100.0
    return score