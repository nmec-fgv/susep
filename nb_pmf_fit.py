#############################################################################
## Calcs parameter estimates for mixed Poisson pmf fit of claim count data ##
## Negative Binomial distribution, MLE using scipy.optimize
#############################################################################

import os
import pickle
from math import log, factorial


filename = 'claim_counts.pkl'
try:
    os.path.exists('/home/ricardob/Susep/Data/' + filename)
    with open('/home/ricardob/Susep/Data/' + filename, 'rb') as file:
        cc_dict = pickle.load(file)
except:
    print('File ' + filename + ' not found')


def func_Poi_month(mmmaa):
