#############################################################################
## Calcs parameter estimates for mixed Poisson pmf fit of claim count data ##
##        Negative Binomial distribution, MLE using scipy.optimize         ##
#############################################################################

import os
import pickle
from math import log, factorial
import numpy as np
from scipy.optimize import minimize


def fun_NegBin(x):
    '''
    Log-likelihood of Negative Binomially distributed count of claims
    Parameters: x[0] -> a
                x[1] -> lambda
    '''
    
    term1 = 0
    term3 = 0
    for i in range(len(k)):
        for j in range(k[i]):
            term1 += log(x[0] + j)
        term3 += (x[0] + k[i]) * log(x[0] + x[1] * d[i])
    
    term2 = len(k) * x[0] * log(x[0])
    term4 = log(x[1]) * sum(k)
    res = - term1 - term2 + term3 - term4
    return res

def fun_NegBin_grad(x):
    '''
    Gradient of log-likelihood of Negative Binomial
    Parameters: x[0] -> a
                x[1] -> lambda
    '''

    grad = np.zeros(2)
    term01 = 0
    term03 = 0
    term04 = 0
    term11 = 0
    for i in range(len(k)):
        for j in range(k[i]):
            term01 += (1 / (x[0] + j))
        term03 += log(x[0] + x[1] * d[i])
        term04 += (x[0] + k[i])/(x[0] + x[1] * d[i])
        term11 += d[i] * (x[0] + k[i])/(x[0] + x[1] * d[i])

    term02 = len(k) * log(x[0]) + len(k)
    term12 = (1 / x[1]) * sum(k)
    grad[0] = - term01 - term02 + term03 + term04
    grad[1] = term11 - term12
    return grad

def initguess_lambda():
    '''Initial guess for lambda parameter'''

    lambda_numerator = 0
    lambda_denominator = 0
    for item in cc_data:
        for i, value in enumerate(item[0].values()):
            if i > 0:
                lambda_numerator += i * value

        lambda_denominator += sum(item[2].values())

    if lambda_denominator > 0:
        lambda_ig = lambda_numerator / lambda_denominator
    else:
        lambda_ig = 0

    return lambda_ig

def initguess_a():
    '''Initial guess for overdispersion parameter a'''

    term1 = 0 
    term2 = 0
    for i in range(len(k)):
        term1 += (k[i] - lambda_ig * d[i])**2 - lambda_ig * d[i]
        term2 += (lambda_ig * d[i])**2
    
    a_ig = term2 / term1
    return a_ig


if __name__ == '__main__':
    
    years = ('07', '08', '09', '10', '11', '12', '13', '14')
    years2 = ('08', '09', '10', '11')
    months = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')
    quarters = ('1T', '2T', '3T', '4T')

    nb_month = {}
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
            
            k = np.array(cc_data[0][3])
            d = np.array(cc_data[0][4])
            lambda_ig = initguess_lambda()
            x0 = np.array([initguess_a(), lambda_ig])
            nb_mle = minimize(fun_NegBin, x0, method='TNC', jac=fun_NegBin_grad, bounds=((1e-4, None), (1e-4, None)), options={'disp': True})
            print('Data: ' + mmm + aa + '; x0: ', x0, 'final estimates: ', nb_mle.x)
            nb_month[mmm+aa] = nb_mle.x 
    
    nb_quarter = {}
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
                
            k = np.array(cc_data[0][3])
            d = np.array(cc_data[0][4])
            for i in range(1, len(months_qtr)):
                k = np.append(k, np.array(cc_data[i][3]))
                d = np.append(d, np.array(cc_data[i][4]))
            lambda_ig = initguess_lambda()
            x0 = np.array([initguess_a(), lambda_ig])
            nb_mle = minimize(fun_NegBin, x0, method='TNC', jac=fun_NegBin_grad, bounds=((1e-4, None), (1e-4, None)), options={'disp': True})
            print('Data: ' + tri + aa + '; x0: ', x0, 'final estimates: ', nb_mle.x)
            nb_quarter[tri+aa] = nb_mle.x 

    nb_year = {}
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
            
        k = np.array(cc_data[0][3])
        d = np.array(cc_data[0][4])
        for i in range(1, len(months)):
            k = np.append(k, np.array(cc_data[i][3]))
            d = np.append(d, np.array(cc_data[i][4]))
        lambda_ig = initguess_lambda()
        x0 = np.array([initguess_a(), lambda_ig])
        nb_mle = minimize(fun_NegBin, x0, method='TNC', jac=fun_NegBin_grad, bounds=((1e-4, None), (1e-4, None)), options={'disp': True})
        print('Data: ' + aa + '; x0: ', x0, 'final estimates: ', nb_mle.x)
        nb_year[aa] = nb_mle.x 

    nb_global = {}
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
            
    k = np.array(cc_data[0][3])
    d = np.array(cc_data[0][4])
    for i in range(1, len(years2)*len(months)):
        k = np.append(k, np.array(cc_data[i][3]))
        d = np.append(d, np.array(cc_data[i][4]))
    lambda_ig = initguess_lambda()
    x0 = np.array([initguess_a(), lambda_ig])
    nb_mle = minimize(fun_NegBin, x0, method='TNC', jac=fun_NegBin_grad, bounds=((1e-4, None), (1e-4, None)), options={'disp': True})
    print('Data: global; x0: ', x0, 'final estimates: ', nb_mle.x)
    nb_global['global'] = nb_mle.x 

    est_nb = {}
    est_nb['NegBin_month'] = nb_month
    est_nb['NegBin_quarter'] = nb_quarter
    est_nb['NegBin_year'] = nb_year
    est_nb['NegBin_global'] = nb_global
    try:
        os.remove('Data/est_negbin_car.pkl')
    except OSError:
        pass

    with open('Data/est_negbin_car.pkl', 'wb') as file:
        pickle.dump(est_nb, file)
