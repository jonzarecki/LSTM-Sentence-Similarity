import random

from util_files.Constants import dtr, d2, cachedStopWords, model


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
    new_examples = filter(lambda new_ex: check_not_in_dataset(new_ex[0], new_ex[1], data), new_examples)[:5000]
    print "expand_positive_samples added " + str(len(new_examples)) + " new examples"
    return data + new_examples