# coding: utf-8

import pickle
from collections import OrderedDict

from util_files.general_utils import getlayerx


def creatrnnx():
    newp = OrderedDict()
    # print ("Creating neural network")
    newp = getlayerx(newp, '1lstm1', 50, 300, 0.5)
    # newp=getlayerx(newp,'1lstm2',30,50)
    # newp=getlayerx(newp,'1lstm3',40,60)
    # newp=getlayerx(newp,'1lstm4',6)
    # newp=getlayerx(newp,'1lstm5',4)
    newp = getlayerx(newp, '2lstm1', 50, 300, 0.5)
    # newp=getlayerx(newp,'2lstm2',20,10)
    # newp=getlayerx(newp,'2lstm3',10,20)
    # newp=getlayerx(newp,'2lstm4',6)
    # newp=getlayerx(newp,'2lstm5',4)
    # newp=getlayerx(newp,'2lstm3',4)
    # newp['2lstm1']=newp['1lstm1']
    # newp['2lstm2']=newp['1lstm2']
    # newp['2lstm3']=newp['1lstm3']
    return newp
