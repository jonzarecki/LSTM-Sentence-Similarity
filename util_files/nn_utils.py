import numpy
import numpy as np
import theano
from theano import tensor as tensor

from util_files.general_utils import _p, numpy_floatX
from util_files.Constants import options, use_noise, dtr, model


def dropout_layer(state_before, use_noise, rrng, rate):
    proj = tensor.switch(use_noise,
                         (state_before * rrng),
                         state_before * (1 - rate))
    return proj


def getpl2(prevlayer, pre, mymask, used, rrng, size, tnewp):
    proj = lstm_layer2(tnewp, prevlayer, options,
                       prefix=pre,
                       mask=mymask, nhd=size)
    if used:
        print "Added dropout"
        proj = dropout_layer(proj, use_noise, rrng, 0.5)

    return proj


def lstm_layer2(tparams, state_below, options, prefix='lstm', mask=None, nhd=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')].T)
        preact += x_
        preact += tparams[_p(prefix, 'b')]

        i = tensor.nnet.sigmoid(_slice(preact, 0, nhd))
        f = tensor.nnet.sigmoid(_slice(preact, 1, nhd))
        o = tensor.nnet.sigmoid(_slice(preact, 2, nhd))
        c = tensor.tanh(_slice(preact, 3, nhd))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return [h, c]

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')].T) +
                   tparams[_p(prefix, 'b')].T)
    # print "hvals"
    dim_proj = nhd
    [hvals, yvals], updates = theano.scan(_step,
                                          sequences=[mask, state_below],
                                          outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                                     n_samples,
                                                                     dim_proj),
                                                        tensor.alloc(numpy_floatX(0.),
                                                                     n_samples,
                                                                     dim_proj)],
                                          name=_p(prefix, '_layers'),
                                          n_steps=nsteps)
    return hvals


def adadelta(lr, tparams, grads, emb11, mask11, emb21, mask21, y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, (0.95 * rg2 + 0.05 * (g ** 2)))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([emb11, mask11, emb21, mask21, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, (0.95 * ru2 + 0.05 * (ud ** 2)))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def sgd(lr, tparams, grads, emb11, mask11, emb21, mask21, y, cost):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function([emb11, mask11, emb21, mask21, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')
    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, emb11, mask11, emb21, mask21, y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([emb11, mask11, emb21, mask21, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


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
