###########################################
## Poisson regression using statsmodels  ##
###########################################

import os
import pickle
import numpy as np
from statsmodels.discrete.discrete_model import Poisson

mmm = 'jan'
aa = '10'
filename = 'data_mpreg_' + mmm + aa + '.pkl'
try:
    os.path.exists('Data/' + filename)
    with open('Data/' + filename, 'rb') as file:
        data = pickle.load(file)
except:
    print('File ' + filename + ' not found')


