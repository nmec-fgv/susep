#############################################################################
## Calcs parameter estimates for mixed Poisson pmf fit of claim count data ##
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
    '''Estimation and inference of Poisson pmf lambda parameter, monthly data'''

    lambda_numerator = 0
    for i, value in enumerate(cc_dict[mmmaa][0].values()):
        if i > 0:
            lambda_numerator += i * value
    
    lambda_denominator = sum(cc_dict[mmmaa][2].values())
    if lambda_denominator > 0:
        lambda_Poi = lambda_numerator / lambda_denominator
    else:
        lambda_Poi = 0
    
    n = sum(cc_dict[mmmaa][0].values())
    if n > 0:
        CI = (lambda_Poi - 1.96 * (lambda_Poi/n)**0.5, lambda_Poi + 1.96 * (lambda_Poi/n)**0.5)
    else:
        CI = (0, 0)
    
    ll_crossprod = 0
    ll_lnkfactorial = 0
    for i in range(len(cc_dict[mmmaa][3])):
        ll_crossprod += cc_dict[mmmaa][3][i] * cc_dict[mmmaa][4][i]
        ll_lnkfactorial += log(factorial(cc_dict[mmmaa][3][i]))
    
    if lambda_Poi != 0:
        loglikelihood = -lambda_Poi * lambda_denominator + log(lambda_Poi) * lambda_numerator + ll_crossprod - ll_lnkfactorial
    else:
        loglikelihood = 0

    return (lambda_Poi, n, CI, loglikelihood)


def func_Poi_quarter(tri, aa):
    '''Estimation and inference of Poisson pmf lambda parameter, quarterly data'''

    lambda_numerator = 0
    lambda_denominator = 0
    if tri == '1T':
        months = ('jan', 'fev', 'mar')
    elif tri == '2T':
        months = ('abr', 'mai', 'jun')
    elif tri == '3T':
        months = ('jul', 'ago', 'set')
    elif tri == '4T':
        months = ('out', 'nov', 'dez')
    
    n = 0
    ll_crossprod = 0
    ll_lnkfactorial = 0
    for mmm in months:
        for i, value in enumerate(cc_dict[mmm+aa][0].values()):
            if i > 0:
                lambda_numerator += i * value
        
        lambda_denominator += sum(cc_dict[mmm+aa][2].values())
        n += sum(cc_dict[mmm+aa][0].values())
    
        for i in range(len(cc_dict[mmm+aa][3])):
            ll_crossprod += cc_dict[mmm+aa][3][i] * cc_dict[mmm+aa][4][i]
            ll_lnkfactorial += log(factorial(cc_dict[mmm+aa][3][i]))

    if lambda_denominator > 0:
        lambda_Poi = lambda_numerator / lambda_denominator
    else:
        lambda_Poi = 0

    if n > 0:
        CI = (lambda_Poi - 1.96 * (lambda_Poi/n)**0.5, lambda_Poi + 1.96 * (lambda_Poi/n)**0.5)
    else:
        CI = (0, 0)
    
    if lambda_Poi != 0:
        loglikelihood = -lambda_Poi * lambda_denominator + log(lambda_Poi) * lambda_numerator + ll_crossprod - ll_lnkfactorial
    else:
        loglikelihood = 0

    return (lambda_Poi, n, CI, loglikelihood)


if __name__ == '__main__':

    years = ('06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16')
    months = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')
    quarters = ('1T', '2T', '3T', '4T')

    Poi_month = {}
    for aa in years:
        for mmm in months:
            Poi_month[mmm+aa] = func_Poi_month(mmm+aa)
    
    Poi_quarter = {}
    for aa in years:
        for tri in quarters:
            Poi_quarter[tri+aa] = func_Poi_quarter(tri, aa)

    estimators_MixPoi = {}
    estimators_MixPoi['Poi_month'] = Poi_month 
    estimators_MixPoi['Poi_quarter'] = Poi_quarter


    try:
        os.remove('/home/ricardob/Susep/Data/estimators_MixPoi.pkl')
    except OSError:
        pass

    with open('/home/ricardob/Susep/Data/estimators_MixPoi.pkl', 'wb') as file:
        pickle.dump(estimators_MixPoi, file)
