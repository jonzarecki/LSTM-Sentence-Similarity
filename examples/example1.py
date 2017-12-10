from lstm import lstm
from util_files.Constants import data_folder
sls=lstm(data_folder + "bestsem.p",load=True,training=False)

# test=pickle.load(open("semtest.p",'rb'))
#Example
sa="A truly wise man"
sb="He is smart"
print sls.predict_similarity(sa,sb)*4.0+1.0