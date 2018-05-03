##############################################################################
## Regression models of counts and claims data for auto insurance cost eval ##
##############################################################################


import os
import pickle
import numpy as np
from scipy.special import factorial
from scipy.optimize import minimize


def file_load(filename):
    try:
        os.path.exists('Data/' + filename)
        with open('Data/' + filename, 'rb') as file:
            res = pickle.load(file)
    except:
        print('File ' + filename + ' not found')

    return res


class Data:
    '''
    Data preparation for subsequent modeling.
    Loads data from files 'Data/data_mmmaa.pkl' according to period request.
    Returns attribute .data, a dictionary containing 'X_exog', 'y_cas', 'y_rcd', 'y_app' and 'y_out', where y_* is divided in 'y_count' and 'y_claim'.
    
    Parameters:
    ----------
    period, takes 'mmm + aa' string value or '#tr + aa'
    threshold, dict containing keys 'cas', 'rcd', 'app', 'out', which must be provided and set to zero if no threshold is intended.
    '''

    def __init__(self, period, threshold):
        
        periods = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez', '1tr', '2tr', '3tr', '4tr')
        years = ('08', '09', '10', '11')

        if period[:3] not in periods and period[3:] not in years:
            raise Exception('period invalid or outside permissable range')

        for item in threshold.values():
            if isinstance(item, (int, float)) == False:
                raise Exception('threshold invalid, provide permissable dictionary object')

        if period[:3] in periods[:12]:
            (mmm, aa) = (period[:3], period[3:])
            filename = 'data_' + mmm + aa + '.pkl'
            data = file_load(filename)

        elif period[:3] in periods[12:]:
            aux = {}
            if period[0] == '1':
                for i, mmm in enumerate(periods[:3]):
                    filename = 'data_' + mmm + aa + '.pkl'
                    aux[str(i)] = file_load(filename)
            elif period[0] == '2':
                for i, mmm in enumerate(periods[3:6]):
                    filename = 'data_' + mmm + aa + '.pkl'
                    aux[str(i)] = file_load(filename)
            elif period[0] == '3':
                for i, mmm in enumerate(periods[6:9]):
                    filename = 'data_' + mmm + aa + '.pkl'
                    aux[str(i)] = file_load(filename)
            elif period[0] == '4':
                for i, mmm in enumerate(periods[9:12]):
                    filename = 'data_' + mmm + aa + '.pkl'
                    aux[str(i)] = file_load(filename)

            data = {}
            X_list = []
            for i in range(3):
                data['y_cas'] += aux[str(i)]['y_cas']
                data['y_rcd'] += aux[str(i)]['y_rcd']
                data['y_app'] += aux[str(i)]['y_app']
                data['y_out'] += aux[str(i)]['y_out']
                X_list.append(aux[str(i)]['X']
            data['X'] = np.stack(X_list, axis=0)

        def endog_array(y, y_threshold):
            '''
            Internal auxiliary function.
            Takes y list argument and computes arrays of counts and claims.
            '''

            if y_threshold == 0:
                y_count = np.empty([len(y)])
                y_claim = np.empty([len(y)])
                for i, item in enumerate(y):
                    y_count[i] = len(item)
                    if len(item) > 0:
                        y_claim[i] = sum(item) / len(item)
                    else:
                        y_claim[i] = 0
                
                return dict([('count', y_count), ('claim', y_claim)])
        
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
        
                return dict([('count_mc', y_count_mc), ('count_ec', y_count_ec), ('claim_mc', y_claim_mc), ('claim_ec', y_claim_ec)])
    
        X_exog = data['X']
        X_exog[:, [64]] = X_exog[:, [64]] / 100
        X_exog[:, 79:83] = X_exog[:, 79:83] / 10000
        y = {}
        y['cas'] = endog_array(data['y_cas'], threshold['cas'])
        y['rcd'] = endog_array(data['y_rcd'], threshold['rcd'])
        y['app'] = endog_array(data['y_app'], threshold['app'])
        y['out'] = endog_array(data['y_out'], threshold['out'])
        self.data = dict([('X_exog', X_exog), ('y', y)])


class Poisson_regress:
    '''
    Provides estimation of Poisson regression model, MLE and PMLE coincide.

    Parameters:
    ----------
    data, must be data attribute previously generated from data class call
    ytype, 2-tuple with values in {'cas', 'rcd', 'app', 'out'} X {'count', 'count_mc', 'count_ec'}    
    '''

    def __init__(self, data, ytype):

        X_exog = data['X_exog']
        y = data['y'][ytype[0]][ytype[1]]

        def log_likelihood(x):
            '''Log-likelihood of Poisson regression model'''
        
            res = np.sum(-y * np.dot(X_exog, x) + np.exp(np.dot(X_exog, x)) + np.log(factorial(y)))
            return res
        
        def gradient(x):
            '''Gradient of Log-likelihood of Poisson regression model'''
        
            aux_vec = -y + np.exp(np.dot(X_exog, x))
            res = (aux_vec[:, np.newaxis] *  X_exog).sum(axis=0)
            return res

        x0 = np.zeros(len(X_exog))
        x0[0] = 1
        x0[1] = np.log(sum(y)/len(y))
        prec_param = 1e-4
        bounds = ((1 - prec_param, 1 + prec_param),)
        for i in range(len(X_exog[0])-1):
            bounds += ((None, None),)
        
        poisson_mle = minimize(LL_Poisson, x0, method='TNC', jac=grad_LL_Poisson, bounds=bounds, options={'disp': True})
        if poisson_mle.success == 0
            poisson_mle = minimize(LL_Poisson, x0, method='L-BFGS-B', jac=grad_LL_Poisson, bounds=bounds, options={'disp': True})


    def var_MLH(x):
        '''Variance for Poisson MLE using Hessian'''
    
        sum_res = 0
        for i in range(len(y)):
            sum_res += np.exp(np.dot(X_exog[i], x)) * np.outer(X_exog[i], X_exog[i])
            return np.linalg.inv(sum_res)
    
    
    def var_MLOP(x):
        '''Variance for Poisson MLE using summed outer product of first derivatives'''
    
        sum_res = 0
        for i in range(len(y)):
            sum_res += (y[i] - np.exp(np.dot(X_exog[i], x)))**2 * np.outer(X_exog[i], X_exog[i])
            return np.linalg.inv(sum_res)



if __name__ == '__main__':
      
    threshold = dict([('y_cas', 0), ('y_rcd', 0), ('y_app', 0), ('y_out', 0)])
    jan08 = Data('jan08', threshold)
    poi_cas_jan08 = Poisson_regress(jan08.data, ('cas', 'count')) # use shelve to archive


#            print('Poisson MLE estimates: ', poisson_mle.x)
#            try:
#                os.remove('Data/PoiMLE_cas-count_' + mmm + aa + '.pkl')
#            except OSError:
#                pass
#                    
#            with open('Data/PoiMLE_cas-count_' + mmm + aa + '.pkl', 'wb') as file:
#                pickle.dump(poisson_mle, file)
#            
#            print('File PoiMLE_cas-count_' + mmm + aa + '.pkl saved') 
