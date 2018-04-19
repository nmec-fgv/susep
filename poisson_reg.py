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

endog = np.array([])
exog = np.array([])
exposure = np.array([])
for i in data:
    endog = np.append(endog, i[-2])
    exog = np.append(exog, [1] + i[:-2])
    exposure = np.append(exposure, -1)

results = Poisson(endog, exog, exposure=exposure)
results.fit()
