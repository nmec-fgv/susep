##############################################################################
## Regression models of counts and claims data for auto insurance cost eval ##
##############################################################################


import os
import pickle
import numpy as np
import shelve
from scipy.special import factorial
from scipy.optimize import minimize


def file_load(filename):
    try:
        os.path.exists('/home/pgsqldata/Susep/' + filename)
        with open('/home/pgsqldata/Susep/' + filename, 'rb') as file:
            x = pickle.load(file)
    except:
        print('File ' + filename + ' not found')

    return x


class Data:
    '''
    Data preparation for subsequent modeling.
    Loads data from files 'data_mmmaa.pkl' according to period request.
    Returns X_exog', and 'y' according to data type requested ({'cas', 'rcd', 'app', 'out'} X {'claim', 'count'}).
    
    Parameters:
    ----------
    period, takes 'mmm' string value or '#tr'
    threshold, dict containing keys 'cas', 'rcd', 'app', 'out', which must be provided and set to zero if no threshold is intended.
    '''

    def __init__(self, period, aa, dtype):
        
        periods = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez', '1tr', '2tr', '3tr', '4tr')
        years = ('08', '09', '10', '11')

        if period not in periods and aa not in years:
            raise Exception('period invalid or outside permissible range')

        if period in periods[:12]:
            mmm = period
            filename = 'data_' + mmm + aa + '.pkl'
            data = file_load(filename)
            data['X'] = data['X'].tolist()
            data['y'] = data['y_'+dtype[0]]

        elif period in periods[12:]:
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
            data['y'] = []
            data['X'] = []
            for i in range(3):
                data['y'] += aux[str(i)]['y_'+dtype[0]]
                for item in aux[str(i)]['X']:
                    data['X'].append(item)

        if dtype[1] == 'count':
            def convert_arr(X_in, y_in):
                '''
                Internal auxiliary function.
                Takes X, y list argument and computes arrays for counts.
                '''
    
                y = []
                X = X_in
                for i, item in enumerate(y_in):
                    y.append(len(item))

                return (X, y)

        elif dtype[1] == 'claim':
            def convert_arr(X_in, y_in):
                '''
                Internal auxiliary function.
                Takes X, y list argument and computes arrays for claims.
                '''
    
                y = []
                X = []
                for i, item in enumerate(y_in):
                    if len(item) > 0:
                        for j in range(len(item)):
                            y.append(item[j])
                            X.append(X_in[i])
                    else:
                        y.append(0)
                        X.append(X_in[i])

                return (X, y)

        (X, y) = convert_arr(data['X'], data['y'])
        y = np.asarray(y)
        X = np.asarray(X)
        X[:, [64]] = X[:, [64]] / 100
        X[:, 79:83] = X[:, 79:83] / 100000
        self.X = X
        self.y = y
    
    def threshold(self, threshold):
        pass

    def desc_stats(self):
        pass

class Poisson(Data):
    '''
    Provides estimation of Poisson regression model, MLE and PMLE coincide.

    Parameters:
    ----------
    data, must be data attribute previously generated from data class call
    ytype, 2-tuple with values in {'cas', 'rcd', 'app', 'out'} X {'count', 'count_mc', 'count_ec'}    
    '''

    def __init__(self, period, aa, dtype, threshold=None):
        super().__init__(period, aa, dtype)
        
        if threshold == None:
            X_exog = self.X
            y = self.y

        def log_likelihood(x):
            '''Log-likelihood of Poisson regression model'''
        
            res = np.sum(-y * np.dot(X_exog, x) + np.exp(np.dot(X_exog, x)) + np.log(factorial(y)))
            return res
        
        def gradient(x):
            '''Gradient of Log-likelihood of Poisson regression model'''
        
            aux_vec = -y + np.exp(np.dot(X_exog, x))
            res = (aux_vec[:, np.newaxis] *  X_exog).sum(axis=0)
            return res

        x0 = np.zeros(len(X_exog[0]))
        x0[0] = 1
        x0[1] = np.log(sum(y)/len(y))
        prec_param = 1e-4
        bounds = ((1 - prec_param, 1 + prec_param),)
        for i in range(len(X_exog[0])-1):
            bounds += ((None, None),)
        
        res = minimize(log_likelihood, x0, method='TNC', jac=gradient, bounds=bounds, options={'disp': True})
        if res.success == 0:
            res = minimize(log_likelihood, x0, method='L-BFGS-B', jac=gradient, bounds=bounds, options={'disp': True})
            if res.success == 0:
                res = minimize(log_likelihood, x0, method='SLSQP', jac=gradient, bounds=bounds, options={'disp': True})

        self.fit = res


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
#    periods = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez', '1tr', '2tr', '3tr', '4tr')
#    periods = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')
    periods = ('1tr',)
#    years = ('08', '09', '10', '11')
    years = ('08',)
    for period in periods:
        for aa in years:
            dtype = ('rcd', 'count')
            x = Poisson(period, aa, dtype)
