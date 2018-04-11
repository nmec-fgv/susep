#############################################
## Count data extraction from pickle files ##
#############################################

import os
import pickle

filename = freq_data_mmmaa.pkl

try:
    os.path.exists('/home/ricardob/Susep/Data/' + filename)
    with open('/home/ricardob/Susep/Data/' + filename, 'rb') as file:
        data = pickle.load(file)
except:
    print('File ' + filename + ' not found')
