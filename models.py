############################################################################
## Regressions of count and claim data for auto-insurance cost estimation ##
############################################################################

#Todo: class Regression
#      init y
#      class Poisson

import os
import pickle
import numpy as np
from math import exp, log, factorial
from scipy.optimize import minimize


def LL_Poisson(x):
    '''Log-likelihood of Poisson regression model'''

    sum_res = 0
    for i in range(len(y)):
        sum_res += -y[i] * np.dot(X_exog[i], x) + exp(np.dot(X_exog[i], x)) + log(factorial(y[i]))
    return sum_res


def grad_LL_Poisson(x):
    '''Gradient of Log-likelihood of Poisson regression model'''

    sum_res = 0
    for i in range(len(y)):
        sum_res += -(y[i] - exp(np.dot(X_exog[i], x))) *  X_exog[i]
    return sum_res


def var_Poisson_MLH(x):
    '''Variance for Poisson MLE using Hessian'''

    sum_res = 0
    for i in range(len(y)):
        sum_res += exp(np.dot(X_exog[i], x)) * np.outer(X_exog[i], X_exog[i])
        return np.linalg.inv(sum_res)


def var_Poisson_MLOP(x):
    '''Variance for Poisson MLE using summed outer product of first derivatives'''

    sum_res = 0
    for i in range(len(y)):
        sum_res += (y[i] - exp(np.dot(X_exog[i], x)))**2 * np.outer(X_exog[i], X_exog[i])
        return np.linalg.inv(sum_res)


def endog_array(y, y_threshold=None):
    '''Takes y list argument and computes arrays of counts and claims'''

    if y_threshold == None:
        y_count = np.empty([len(y)])
        y_claim = np.empty([len(y)])
        for i, item in enumerate(y):
            y_count[i] = len(item)
            if len(item) > 0:
                y_claim[i] = sum(item) / len(item)
            else:
                y_claim[i] = 0
        
        return dict([('y_count', y_count), ('y_claim', y_claim)])

    else:
        y_count_mc = np.empty([len(y)])
        y_count_ec = np.empty([len(y)])
        y_claim_mc = np.empty([len(y)])
        y_claim_ec = np.empty([len(y)])
        for i, item in enumerate(y):
            y_count_mc[i] = sum(x < y_threshold for x in item)
            y_count_ec[i] = sum(x >= y_threshold for x in item)
            if len([x for x in item if x < y_threshold]) > 0:
                y_claim_mc[i] = sum([x for x in item if x < y_threshold])/ len([x for x in item if x < y_threshold])
            else:
                y_claim_mc[i] = 0
            if len([x for x in item if x >= y_threshold]) > 0:
                y_claim_ec[i] = sum([x for x in item if x >= y_threshold])/ len([x for x in item if x >= y_threshold])
            else:
                y_claim_ec[i] = 0

        return dict([('y_count_mc', y_count_mc), ('y_count_ec', y_count_ec), ('y_claim_mc', y_claim_mc), ('y_claim_ec', y_claim_ec)])


if __name__ == '__main__':
    mmm = 'jan'
    aa = '08'
    filename = 'data_' + mmm + aa + '.pkl'
    try:
        os.path.exists('Data/' + filename)
        with open('Data/' + filename, 'rb') as file:
            data = pickle.load(file)
    except:
        print('File ' + filename + ' not found')
    
    X_exog = data['X']
    X_exog[:, [64]] = X_exog[:, [64]] / 100
    X_exog[:, 79:83] = X_exog[:, 79:83] / 1000000
    y0 = data['y_rcd']
    y_dict = endog_array(y0)
    y = y_dict['y_count']

    filename = 'PoiMLE_initguess.pkl'
    try:
        os.path.exists(filename)
        with open(filename, 'rb') as file:
            x0 = pickle.load(file)
    except:
        print('File ' + filename + ' not found')
    x0[0] = 1
    prec_param = 1e-4
    bounds = ((1 - prec_param, 1 + prec_param),)
    for i in range(len(X_exog[0])-1):
        bounds += ((None, None),)
    
#    poisson_mle1 = minimize(LL_Poisson, x0, method='L-BFGS-B', jac=grad_LL_Poisson, bounds=bounds, options={'disp': True})
#    poisson_mle2 = minimize(LL_Poisson, x0, method='SLSQP', jac=grad_LL_Poisson, bounds=bounds, options={'disp': True})
    poisson_mle3 = minimize(LL_Poisson, x0, method='TNC', jac=grad_LL_Poisson, bounds=bounds, options={'disp': True})
#    print('Poisson MLE estimates: ', poisson_mle1.x)
#    print('Poisson MLE estimates: ', poisson_mle2.x)
    print('Poisson MLE estimates: ', poisson_mle3.x)
#    var_poisson_mlh = var_Poisson_MLH(poisson_mle1.x)
#    var_poisson_mlop = var_Poisson_MLOP(poisson_mle1.x)
#    print('Poisson Var MLH estimate: ', var_poisson_mlh)
#    print('Poisson Var MLOP estimate: ', var_poisson_mlop)

    try:
        os.remove('Data/PoiMLE_est_' + mmm + aa + '.pkl')
    except OSError:
        pass
            
    with open('Data/PoiMLE_est_' + mmm + aa + '.pkl', 'wb') as file:
        pickle.dump(poisson_mle3, file)
    
    print('File data_' + mmm + aa + '.pkl saved') 
