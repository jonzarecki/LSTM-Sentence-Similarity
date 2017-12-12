import random

from util_files.data_utils import sentence_unigram_probability


def weighted_choice_sub(weights):
    rnd = random.random() * sum(weights)
    for i, w in enumerate(weights):
        rnd -= w
        if rnd < 0:
            return i


def build_negative_sample(curr_sent, sent_prob):
    sent_prob_pairs = list(sent_prob.iteritems())
    rand_idx = weighted_choice_sub(sent_prob.itervalues())
    neg_sent, _ = sent_prob_pairs[rand_idx]
    return [curr_sent, neg_sent, 0.0]  # the most negative score is 0.0 (where the range in the data is 0-5)


def build_sent_probability_dict(data):
    sent_prob = dict()
    total_prob = 0
    for sent1, sent2, score in data:
        sent_prob[sent1] = sentence_unigram_probability(sent1)
        sent_prob[sent2] = sentence_unigram_probability(sent2)
        total_prob += sent_prob[sent1] + sent_prob[sent2]

    for sent, prob in sent_prob.iteritems():  # normalize probabilities
        sent_prob[sent] = prob / total_prob

    return sent_prob


def extend_negative_samples(data):
    sent_prob = build_sent_probability_dict(data)
    for sent in random.sample(sent_prob.iterkeys(), 7000):  # build new negative samples
        data.append(build_negative_sample(sent, sent_prob))
    return data