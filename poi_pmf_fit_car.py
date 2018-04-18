#############################################################################
## Calcs parameter estimates for mixed Poisson pmf fit of claim count data ##
#############################################################################

import os
import pickle
from math import log, factorial


def func_Poisson():
    '''Estimation and inference of Poisson pmf lambda parameter'''

    lambda_numerator = 0
    lambda_denominator = 0
    n = 0
    ll_crossprod = 0
    ll_lnkfactorial = 0
    for item in cc_data:
        for i, value in enumerate(item[0].values()):
            if i > 0:
                lambda_numerator += i * value
        
        lambda_denominator += sum(item[2].values())
        n += sum(item[0].values())
    
        for i in range(len(item[3])):
            if item[4][i] > 0:
                ll_crossprod += item[3][i] * log(item[4][i])
            ll_lnkfactorial += log(factorial(item[3][i]))

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
    years2 = ('08', '09', '10', '11')
    months = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')
    quarters = ('1T', '2T', '3T', '4T')

    Poi_month = {}
    for aa in years:
        for mmm in months:
            cc_data = []
            filename = 'cc_car_' + mmm + aa + '.pkl'
            try:
                os.path.exists('Data/' + filename)
                with open('Data/' + filename, 'rb') as file:
                    cc_data.append(pickle.load(file))
            except:
                print('File ' + filename + ' not found')

            Poi_month[mmm+aa] = func_Poisson()
    
    Poi_quarter = {}
    for aa in years:
        for tri in quarters:
            if tri == '1T':
                months_qtr = ('jan', 'fev', 'mar')
            elif tri == '2T':
                months_qtr = ('abr', 'mai', 'jun')
            elif tri == '3T':
                months_qtr = ('jul', 'ago', 'set')
            elif tri == '4T':
                months_qtr = ('out', 'nov', 'dez')

            cc_data = []
            for mmm in months_qtr:
                filename = 'cc_car_' + mmm + aa + '.pkl'
                try:
                    os.path.exists('Data/' + filename)
                    with open('Data/' + filename, 'rb') as file:
                        cc_data.append(pickle.load(file))
                except:
                    print('File ' + filename + ' not found')
            Poi_quarter[tri+aa] = func_Poisson()
    
    Poi_year = {}
    for aa in years:
        cc_data = []
        for mmm in months:
            filename = 'cc_car_' + mmm + aa + '.pkl'
            try:
                os.path.exists('Data/' + filename)
                with open('Data/' + filename, 'rb') as file:
                    cc_data.append(pickle.load(file))
            except:
                print('File ' + filename + ' not found')
        
        Poi_year[aa] = func_Poisson()

    Poi_global = {}
    cc_data = []
    for aa in years2:
        for mmm in months:
            filename = 'cc_car_' + mmm + aa + '.pkl'
            try:
                os.path.exists('Data/' + filename)
                with open('Data/' + filename, 'rb') as file:
                    cc_data.append(pickle.load(file))
            except:
                print('File ' + filename + ' not found')
        
    Poi_global['global'] = func_Poisson()

    est_poi = {}
    est_poi['Poisson_month'] = Poi_month 
    est_poi['Poisson_quarter'] = Poi_quarter
    est_poi['Poisson_year'] = Poi_year
    est_poi['Poisson global'] = Poi_global
    try:
        os.remove('Data/est_poisson_car.pkl')
    except OSError:
        pass

    with open('Data/est_poisson_car.pkl', 'wb') as file:
        pickle.dump(est_poi, file)
