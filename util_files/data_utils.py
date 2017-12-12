import random
import numpy
import numpy as np

from util_files.Constants import dtr, d2, cachedStopWords, model, word_prob, total_counts


def pfl(s):
    for i in dtr['syn'][0]:
        s.append(i)
    return s


def chsyn(sent, trn_data, ignore_flag):
    from util_files.Constants import flg
    cnt = 0
    sent_wrods = sent.split()
    sent_words = sent.split()

    for i in sent_wrods:
        sent_words.append(i)
    for i in range(0, len(sent_words)):
        q = sent_words[i]
        mst = ''
        if q not in d2:
            continue
        if flg == 1 and not ignore_flag:
            trn_data = pfl(trn_data)
            flg = 0

        if q in cachedStopWords or q.title() in cachedStopWords or q.lower() in cachedStopWords:
            # print q,"skipped"
            continue
        if q in d2 or q.lower() in d2:
            if q in d2:
                mst = findsim(q)
            # print q,mst
            elif q.lower() in d2:
                mst = findsim(q)
            if q not in model:
                continue

        if mst in model:
            if q == mst:
                mst = ''

                continue
            if model.similarity(q, mst) < 0.6:
                continue
            # print sent_words[i],mst
            sent_words[i] = mst
            if q.find('ing') != -1:
                if sent_words[i] + 'ing' in model:
                    sent_words[i] += 'ing'
                if sent_words[i][:-1] + 'ing' in model:
                    sent_words[i] = sent_words[i][:-1] + 'ing'
            if q.find('ed') != -1:
                if sent_words[i] + 'ed' in model:
                    sent_words[i] += 'ed'
                if sent_words[i][:-1] + 'ed' in model:
                    sent_words[i] = sent_words[i][:-1] + 'ed'
            cnt += 1
    return ' '.join(sent_words), cnt


def findsim(wd):
    syns = d2[wd]
    x = random.randint(0, len(syns) - 1)
    return syns[x]


def check_not_in_dataset(sa, sb, data):
    for i in data:
        if sa == i[0] and sb == i[1]:
            return False
        if sa == i[1] and sb == i[0]:
            return False
    return True  # don't apear already


def expand_positive_examples(data, ignore_flag):
    new_examples = []
    for m in range(0, 10):
        for ex in data:
            sa, cnt1 = chsyn(ex[0], data, ignore_flag)
            sb, cnt2 = chsyn(ex[1], data, ignore_flag)
            if cnt1 > 0 and cnt2 > 0:
                new_ex = [sa, sb, ex[2]]
                new_examples.append(new_ex)
    new_examples = filter(lambda new_ex: check_not_in_dataset(new_ex[0], new_ex[1], data), new_examples)
    print "expand_positive_samples added " + str(len(new_examples)) + " new examples"
    return data + new_examples


def prepare_data(data):
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
    emb1, mas1 = getmtr(xa1, maxlen)
    emb2, mas2 = getmtr(xb1, maxlen)

    y2 = np.array(y2, dtype=np.float32)

    assert len(data) == len(emb1), "_prepare_embeddings assertion broken"
    return emb1, mas1, emb2, mas2, y2


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


def sentence_unigram_probability(sent):
    """ pretty weak language model but should be enough"""
    prob = 1
    for word in sent.split():
        if word in word_prob:
            prob *= word_prob[word]
        else:
            prob *= 1.0 / total_counts
    return prob