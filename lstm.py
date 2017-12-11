# coding: utf-8
from random import *
import theano.tensor as T
from theano import config
import theano
import numpy as np
import scipy.stats as meas
from collections import OrderedDict
import time
import theano.tensor as tensor
import pickle
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from util_files.Constants import use_noise
from util_files.nn_utils import getpl2, adadelta, embed_sentence
from util_files.general_utils import getlayerx, init_tparams
from util_files.data_utils import prepare_data


def creatrnnx():
    """ builds an orderedDict for saving the model into disk """
    newp = OrderedDict()
    # print ("Creating neural network")
    newp = getlayerx(newp, '1lstm1', 50, 300, 1.5)
    # newp=getlayerx(newp,'1lstm2',30,50)
    # newp=getlayerx(newp,'1lstm3',40,60)
    # newp=getlayerx(newp,'1lstm4',6)
    # newp=getlayerx(newp,'1lstm5',4)
    newp = getlayerx(newp, '2lstm1', 50, 300, 1.5)
    # newp=getlayerx(newp,'2lstm2',20,10)
    # newp=getlayerx(newp,'2lstm3',10,20)
    # newp=getlayerx(newp,'2lstm4',6)
    # newp=getlayerx(newp,'2lstm5',4)
    # newp=getlayerx(newp,'2lstm3',4)
    # newp['2lstm1']=newp['1lstm1']
    # newp['2lstm2']=newp['1lstm2']
    # newp['2lstm3']=newp['1lstm3']
    return newp


class lstm():
    def __init__(self, nam, load=False, training=False):
        if load:
            newp = pickle.load(open(nam, 'rb'))  # nam is only used here
        else:
            newp = creatrnnx()
        tnewp = init_tparams(newp)
        for i in newp.keys():
            if i[0] == '1':
                newp['2' + i[1:]] = newp[i]
        y = tensor.vector('y', dtype=config.floatX)
        mask11 = tensor.matrix('mask11', dtype=config.floatX)
        mask21 = tensor.matrix('mask21', dtype=config.floatX)
        emb11 = theano.tensor.ftensor3('emb11')
        emb21 = theano.tensor.ftensor3('emb21')
        trng = RandomStreams(1234)

        rate = 0.5
        rrng = trng.binomial(emb11.shape, p=1 - rate, n=1, dtype=emb11.dtype)

        proj11 = getpl2(emb11, '1lstm1', mask11, False, rrng, 50, tnewp)[-1]
        proj21 = getpl2(emb21, '2lstm1', mask21, False, rrng, 50, tnewp)[-1]
        dif = (proj21 - proj11).norm(L=1, axis=1)
        s2 = T.exp(-dif)
        sim = T.clip(s2, 1e-7, 1.0 - 1e-7)
        lr = tensor.scalar(name='lr')
        ys = T.clip((y - 1.0) / 4.0, 1e-7, 1.0 - 1e-7)
        cost = T.mean((sim - ys) ** 2)
        ns = emb11.shape[1]
        self.f2sim = theano.function([emb11, mask11, emb21, mask21], sim, allow_input_downcast=True)
        self.f_proj11 = theano.function([emb11, mask11], proj11, allow_input_downcast=True)
        self.f_cost = theano.function([emb11, mask11, emb21, mask21, y], cost, allow_input_downcast=True)  # not used

        if training:
            gradi = tensor.grad(cost, wrt=tnewp.values())  # /bts
            grads = []
            l = len(gradi)
            for i in range(0, l / 2):
                gravg = (gradi[i] + gradi[i + l / 2]) / (4.0)
                grads.append(gravg)
            for i in range(0, len(tnewp.keys()) / 2):
                grads.append(grads[i])

            self.f_grad_shared, self.f_update = adadelta(lr, tnewp, grads, emb11, mask11, emb21, mask21, y, cost)

    @staticmethod
    def _prepare_embeddings(x1, x2):
        ls = []
        ls2 = []
        for j in range(0, len(x1)):
            ls.append(embed_sentence(x1[j]))
            ls2.append(embed_sentence(x2[j]))
        trconv = np.dstack(ls)
        trconv2 = np.dstack(ls2)
        emb2 = np.swapaxes(trconv2, 1, 2)
        emb1 = np.swapaxes(trconv, 1, 2)
        return emb1, emb2

    def train_lstm(self, train, max_epochs, batch_size=32, disp_freq=40, lrate=0.0001):
        print "train_lstm - Start Training"
        batch_count = 0
        for eidx in xrange(0, max_epochs):
            sta = time.time()
            print 'Epoch', eidx
            rnd_order = sample(xrange(len(train)), len(train))  # random order for training each batch
            for batch_start_idx in range(0, len(train), batch_size):
                batch_count += 1

                batch_train = []
                batch_end = batch_start_idx + batch_size if batch_end <= len(train) else len(train)
                for z in range(batch_start_idx, batch_end):
                    batch_train.append(train[rnd_order[z]])  # extract from the training file
                # batch_train=train[batch_start_idx:batch_start_idx+32]
                # shuffle(batch_train)
                x1, mas1, x2, mas2, y2 = prepare_data(batch_train)
                use_noise.set_value(1.)
                emb1, emb2 = self._prepare_embeddings(x1, x2)

                cost = self.f_grad_shared(emb2, mas2, emb1, mas1, y2)
                s = self.f_update(lrate)

                if np.mod(batch_count, disp_freq) == 0:
                    print 'Epoch ', eidx, 'Update ', batch_count, 'Cost ', cost
            sto = time.time()
            print "epoch took:", sto - sta

    def chkterr2(self, mydata):
        num = len(mydata)
        px = []
        yx = []
        use_noise.set_value(0.)
        for i in range(0, num, 256):
            q = []
            x = i + 256
            if x > num:
                x = num
            for j in range(i, x):
                q.append(mydata[j])
            x1, mas1, x2, mas2, y2 = prepare_data(q)
            emb1, emb2 = self._prepare_embeddings(x1, x2)
            pred = (self.f2sim(emb1, mas1, emb2, mas2)) * 4.0 + 1.0
            # dm1=np.ones(mas1.shape,dtype=np.float32)
            # dm2=np.ones(mas2.shape,dtype=np.float32)
            # corr=f_cost(emb1,mas1,emb2,mas2,y2)
            for z in range(0, len(q)):
                yx.append(y2[z])
                px.append(pred[z])
        # count.append(corr)
        px = np.array(px)
        yx = np.array(yx)
        # print "average error= "+str(np.mean(acc))
        return np.mean(np.square(px - yx)), meas.pearsonr(px, yx)[0], meas.spearmanr(yx, px)[0]

    def predict_similarity(self, sa, sb):
        q = [[sa, sb, 0]]
        x1, mas1, x2, mas2, y2 = prepare_data(q)
        assert len(x1) == len(q), "ASdasd"
        use_noise.set_value(0.)
        emb1, emb2 = self._prepare_embeddings(x1, x2)

        return self.f2sim(emb1, mas1, emb2, mas2)
