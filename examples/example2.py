from lstm import lstm
import pickle
from util_files.Constants import data_folder, models_folder

model_name = "bestsem.p"
sls=lstm(models_folder + model_name, load=True)
print model_name

test = pickle.load(open(data_folder + "semtest.p",'rb'))
print sls.check_error(test) #Mean Squared Error,Pearson, Spearman
