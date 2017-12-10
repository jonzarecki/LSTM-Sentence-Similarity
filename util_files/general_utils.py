from collections import OrderedDict

import numpy as np
import theano
from theano import config


def _p(pp, name):
    return '%s_%s' % (pp, name)


def numpy_floatX(data):
    """ converts $data to numpy arrays"""
    return np.asarray(data, dtype=config.floatX)


def genm(mu, sigma, n1, n2):
    return np.random.normal(mu, sigma, (n1, n2))


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def getlayerx(d, pref, n, nin, bn_val):
    mu = 0.0
    sigma = 0.2
    U = np.concatenate([genm(mu, sigma, n, n), genm(mu, sigma, n, n), genm(mu, sigma, n, n),
                        genm(mu, sigma, n, n)]) / np.sqrt(n)
    U = np.array(U, dtype=np.float32)
    W = np.concatenate([genm(mu, sigma, n, nin), genm(mu, sigma, n, nin), genm(mu, sigma, n, nin),
                        genm(mu, sigma, n, nin)]) / np.sqrt(np.sqrt(n * nin))
    W = np.array(W, dtype=np.float32)

    d[_p(pref, 'U')] = U
    # b = numpy.zeros((n * 300,))+1.5
    b = np.random.uniform(-0.5, 0.5, size=(4 * n,))
    b[n:n * 2] = bn_val
    d[_p(pref, 'W')] = W
    d[_p(pref, 'b')] = b.astype(config.floatX)
    return d
