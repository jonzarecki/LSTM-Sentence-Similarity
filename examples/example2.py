from lstm import lstm
import pickle
from util_files.Constants import data_folder, models_folder

model_name = "negative_5000_model.p"
# sls=lstm(models_folder + "bestsem.p",load=True,training=False)
sls=lstm.load_from_pickle_old(models_folder + model_name)
print model_name

test = pickle.load(open(data_folder + "semtest.p",'rb'))
print sls.check_error(test) #Mean Squared Error,Pearson, Spearman
