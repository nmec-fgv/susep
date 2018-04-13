#############################################################################
## Calcs parameter estimates for mixed Poisson pmf fit of claim count data ##
#############################################################################

import os
import pickle


filename = 'claim_counts.pkl'
try:
    os.path.exists('/home/ricardob/Susep/Data/' + filename)
    with open('/home/ricardob/Susep/Data/' + filename, 'rb') as file:
        cc_dict = pickle.load(file)
except:
    print('File ' + filename + ' not found')


def func_lambda_Poi_month(mmmaa):
    '''Estimation of Poisson pmf lambda parameter, monthly data.'''

    lambda_Poi_numerator = 0
    for i, value in enumerate(cc_dict[mmmaa][0].values()):
        if i > 0:
            lambda_Poi_numerator += i * value
    if sum(cc_dict[mmmaa][2].values()) > 0:
        lambda_Poi = lambda_Poi_numerator / sum(cc_dict[mmmaa][2].values())
    else:
        lambda_Poi = 0
    return lambda_Poi


def func_lambda_Poi_quarter(tri, aa):
    '''Estimation of Poisson pmf lambda parameter, quarterly data.'''

    lambda_Poi_numerator = 0
    lambda_Poi_denominator = 0
    if tri == '1T':
        months = ('jan', 'fev', 'mar')
    elif tri == '2T':
        months = ('abr', 'mai', 'jun')
    elif tri == '3T':
        months = ('jul', 'ago', 'set')
    elif tri == '4T':
        months = ('out', 'nov', 'dez')
    for mmm in months:
        for i, value in enumerate(cc_dict[mmm+aa][0].values()):
            if i > 0:
                lambda_Poi_numerator += i * value
        lambda_Poi_denominator += sum(cc_dict[mmm+aa][2].values())
        if lambda_Poi_denominator > 0:
            lambda_Poi = lambda_Poi_numerator / lambda_Poi_denominator
        else:
            lambda_Poi = 0
    return lambda_Poi


if __name__ == '__main__':

    years = ('06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16')
    months = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')
    quarters = ('1T', '2T', '3T', '4T')

    lambda_Poi_month = {}
    for aa in years:
        for mmm in months:
            lambda_Poi_month[mmm+aa] = func_lambda_Poi_month(mmm+aa)
    
    lambda_Poi_quarter = {}
    for aa in years:
        for tri in quarters:
            lambda_Poi_quarter[tri+aa] = func_lambda_Poi_quarter(tri, aa)

    estimators_MixPoi = {}
    estimators_MixPoi['lambda_Poi_month'] = lambda_Poi_month 
    estimators_MixPoi['lambda_Poi_quarter'] = lambda_Poi_quarter


    try:
        os.remove('/home/ricardob/Susep/Data/estimators_MixPoi.pkl')
    except OSError:
        pass

    with open('/home/ricardob/Susep/Data/estimators_MixPoi.pkl', 'wb') as file:
        pickle.dump(estimators_MixPoi, file)
