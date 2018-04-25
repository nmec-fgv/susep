##############################################
## Count data regression using statsmodels  ##
##############################################

import os
import pickle
import numpy as np
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.discrete.discrete_model import NegativeBinomial


mmm = 'jan'
aa = '10'
filename = 'data_mpreg_' + mmm + aa + '.pkl'
try:
    os.path.exists('Data/' + filename)
    with open('Data/' + filename, 'rb') as file:
        data = pickle.load(file)
except:
    print('File ' + filename + ' not found')

data = [item for item in data if item[-1] > 1e-2]

endog = np.empty([len(data)])
exog = np.empty([len(data), 64])
exposure = np.empty([len(data)])
for i, item in enumerate(data):
    endog[i] = item[-2]
    exog[i] = [1] + item[:63]
    exposure[i] = item[-1]

poi_results = Poisson(endog, exog, exposure=exposure)
poi_results2 = poi_results.fit(method='bfgs', maxiter=500)
nb_results = NegativeBinomial(endog, exog, exposure=exposure)
nb_results2 = nb_results.fit(method='bfgs', maxiter=500)
