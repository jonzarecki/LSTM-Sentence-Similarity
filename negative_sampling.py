import random

import copy

from util_files.data_utils import sentence_unigram_probability


def weighted_choice_sub(weights):
    assert sum(weights) != 0, "all weights are 0"
    rnd = random.random() * sum(weights)
    for i, w in enumerate(weights):
        rnd -= w
        if rnd < 0:
            return i

negative_score = 2.0
def build_negative_sample(curr_sent, sent_prob):
    sent_prob_pairs = list(sent_prob.iteritems())
    rand_idx = weighted_choice_sub(list(sent_prob.itervalues()))
    neg_sent, _ = sent_prob_pairs[rand_idx]
    return [curr_sent, neg_sent, negative_score]  # the most negative score is 1.0 (where the range in the data is 1-5)


def build_sent_probability_dict_w2v(data, power=0.75):
    sent_prob = dict()
    for sent_pair in data:
        sent1, sent2 = sent_pair[:2]
        sent_prob[sent1] = sentence_unigram_probability(sent1) ** power  # power of 3/4 like w2v
        sent_prob[sent2] = sentence_unigram_probability(sent2) ** power

    total_prob = sum(sent_prob.itervalues())
    for sent, prob in sent_prob.iteritems():  # normalize probabilities
        sent_prob[sent] = prob / total_prob

    return sent_prob


def build_sent_probability_dict_random(data):
    sent_prob = dict()
    for sent_pair in data:
        sent1, sent2 = sent_pair[:2]
        sent_prob[sent1] = 1.0  # same prob for all
        sent_prob[sent2] = 1.0

    total_prob = sum(sent_prob.itervalues())
    for sent, prob in sent_prob.iteritems():  # normalize probabilities
        sent_prob[sent] = prob / total_prob

    return sent_prob

new_examples_amout = 5000
def extend_negative_samples(data):
    new_data = copy.deepcopy(data)
    sent_prob = build_sent_probability_dict_w2v(new_data)
    sent_pool = list(sent_prob.iterkeys())
    print "neg score ", negative_score
    for sent in [random.sample(sent_pool, 1)[0] for _ in range(new_examples_amout)]:  # new neg samples
        new_data.append(build_negative_sample(sent, sent_prob))
    print "generated " + str(new_examples_amout) + " examples"
    return new_data