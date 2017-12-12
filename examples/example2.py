from lstm import lstm
import pickle
from util_files.Constants import data_folder, models_folder

sls=lstm(models_folder + "bestsem.p",load=True,training=False)
#Gradient compilatoin takes a long time, hence training=False since we're loading examples
test=pickle.load(open(data_folder + "semtest.p",'rb'))
print sls.chkterr2(test) #Mean Squared Error,Pearson, Spearman
