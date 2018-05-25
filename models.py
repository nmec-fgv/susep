#########################################################################
## Regression models of counts and claims data for auto insurance cost ##
#########################################################################


import os
import pickle
import shelve
import numpy as np
import sympy
from scipy.special import factorial
from scipy.optimize import minimize
from scipy.stats import norm
import pdb


# Lower bound parameters for logs and ratios, plus precision parameter:

lb_log = 1e-323
lb_ratio = 1e-308
prec_param = 1e-4

# Auxiliary functions:

def file_load(filename):
    try:
        os.path.exists('/home/pgsqldata/Susep/' + filename)
        with open('/home/pgsqldata/Susep/' + filename, 'rb') as file:
            x = pickle.load(file)
    except:
        print('File ' + filename + ' not found')

    return x

# Classes:

class Data:
    '''
    Data preparation for subsequently running model.
    Loads data from files 'data_mmmaa.pkl' according to period request.
    Returns X', and 'y' according to data type requested ({'cas', 'rcd', 'app', 'out'} X {'claim', 'count'}).
    
    Parameters:
    ----------
    period, takes 'mmm' string value or '#tr'
    threshold, dict containing keys 'cas', 'rcd', 'app', 'out', which must be provided and set to zero if no threshold is intended.
    '''

    def __init__(self, period, aa, dtype, binary_count='no'):
        
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
                if binary_count == 'no':
                    for item in y_in:
                        y.append(len(item))
                elif binary_count == 'yes':
                    for item in y_in:
                        if len(item) > 0:
                            y.append(1)
                        else:
                            y.append(0)
                else:
                    raise Exception('Parameter value for binary_count invalid, must be in {"yes", "no"}.')

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
        X[:,[64]] = X[:,[64]] / 100
        X[:,79:83] = X[:,79:83] / 100000
        X = np.append(X, np.square(X[:, [64]]), axis=1)
        self.X = X
        self.y = y
        self.dtype = dtype


    def desc_stats(self):
        res = {}
        res['nobs'] = len(self.y)
        if self.dtype[1] == 'count':
            (elem, freq) = np.unique(self.y, return_counts=True)
            for i in zip(elem, freq):
                res['count='+str(i[0])] = i[1] / res['nobs']
            res['count_mean'] = np.mean(self.y)
            res['count_std'] = np.std(self.y)
                
        if self.dtype[1] == 'claim':
            res['claims'] = ((self.y[self.y>0.1].mean(), self.y[self.y>0.1].std()), (np.amin(self.y[self.y>0.1]), np.amax(self.y[self.y>0.1])))

        res['X'] = {}
        # Continuous variables
        res['X']['exposure'] = ((np.exp(self.X[:,[0]]).mean(), np.exp(self.X[:,[0]]).std()), (np.amin(np.exp(self.X[:,[0]])), np.amax(np.exp(self.X[:,[0]]))))
        res['X']['idade'] = ((self.X[:,[64]].mean(), self.X[:,[64]].std()), (np.amin(self.X[:,[64]]), np.amax(self.X[:,[64]])))
        res['X']['val_franq'] = ((self.X[:,[79]].mean(), self.X[:,[79]].std()), (self.X[:,[79]][self.X[:,[79]]>1e-5].mean(), self.X[:,[79]][self.X[:,[79]]>1e-5].std()), (np.amin(self.X[:,[79]][self.X[:,[79]]>1e-5]), np.amax(self.X[:,[79]]), len(self.X[:,[79]][self.X[:,[79]]>1e-5]) / res['nobs']))
        res['X']['is_cas'] = ((self.X[:,[80]].mean(), self.X[:,[80]].std()), (self.X[:,[80]][self.X[:,[80]]>1e-5].mean(), self.X[:,[80]][self.X[:,[80]]>1e-5].std()), (np.amin(self.X[:,[80]][self.X[:,[80]]>1e-5]), np.amax(self.X[:,[80]]), len(self.X[:,[80]][self.X[:,[80]]>1e-5]) / res['nobs']))
        res['X']['is_rcd'] = (((self.X[:,[81]].mean(), self.X[:,[81]].std()), (self.X[:,[81]][self.X[:,[81]]>1e-5].mean(), self.X[:,[81]][self.X[:,[81]]>1e-5].std()), (np.amin(self.X[:,[81]][self.X[:,[81]]>1e-5]), np.amax(self.X[:,[81]])), len(self.X[:,[81]][self.X[:,[81]]>1e-5]) / res['nobs']))
        res['X']['is_app'] = (((self.X[:,[82]].mean(), self.X[:,[82]].std()), (self.X[:,[82]][self.X[:,[82]]>1e-5].mean(), self.X[:,[82]][self.X[:,[82]]>1e-5].std()), (np.amin(self.X[:,[82]][self.X[:,[82]]>1e-5]), np.amax(self.X[:,[82]])), len(self.X[:,[82]][self.X[:,[82]]>1e-5]) / res['nobs']))
        # Discrete variables
        res['X']['ano_modelo'] = {}
        res['X']['ano_modelo']['0'] = len(self.X[:,2:10][np.where(~self.X[:,2:10].any(axis=1))[0]]) / res['nobs']
        res['X']['ano_modelo']['1'] = (self.X[:,[2]]==1).sum() / res['nobs']
        res['X']['ano_modelo']['2'] = (self.X[:,[3]]==1).sum() / res['nobs']
        res['X']['ano_modelo']['3'] = (self.X[:,[4]]==1).sum() / res['nobs']
        res['X']['ano_modelo']['4'] = (self.X[:,[5]]==1).sum() / res['nobs']
        res['X']['ano_modelo']['5'] = (self.X[:,[6]]==1).sum() / res['nobs']
        res['X']['ano_modelo']['6-10'] = (self.X[:,[7]]==1).sum() / res['nobs']
        res['X']['ano_modelo']['11-20'] = (self.X[:,[8]]==1).sum() / res['nobs']
        res['X']['ano_modelo']['>20'] = (self.X[:,[9]]==1).sum() / res['nobs']
        res['X']['cod_tarif'] = {}
        res['X']['cod_tarif']['10'] = len(self.X[:,10:23][np.where(~self.X[:,10:23].any(axis=1))[0]]) / res['nobs']
        res['X']['cod_tarif']['11'] = (self.X[:,[10]]==1).sum() / res['nobs']
        res['X']['cod_tarif']['14A'] = (self.X[:,[11]]==1).sum() / res['nobs']
        res['X']['cod_tarif']['14B'] = (self.X[:,[12]]==1).sum() / res['nobs']
        res['X']['cod_tarif']['14C'] = (self.X[:,[13]]==1).sum() / res['nobs']
        res['X']['cod_tarif']['15'] = (self.X[:,[14]]==1).sum() / res['nobs']
        res['X']['cod_tarif']['16'] = (self.X[:,[15]]==1).sum() / res['nobs']
        res['X']['cod_tarif']['17'] = (self.X[:,[16]]==1).sum() / res['nobs']
        res['X']['cod_tarif']['18'] = (self.X[:,[17]]==1).sum() / res['nobs']
        res['X']['cod_tarif']['19'] = (self.X[:,[18]]==1).sum() / res['nobs']
        res['X']['cod_tarif']['20'] = (self.X[:,[19]]==1).sum() / res['nobs']
        res['X']['cod_tarif']['21'] = (self.X[:,[20]]==1).sum() / res['nobs']
        res['X']['cod_tarif']['22'] = (self.X[:,[21]]==1).sum() / res['nobs']
        res['X']['cod_tarif']['23'] = (self.X[:,[22]]==1).sum() / res['nobs']
        res['X']['regiao'] = {}
        res['X']['regiao']['01'] = (self.X[:,[23]]==1).sum() / res['nobs']
        res['X']['regiao']['02'] = (self.X[:,[24]]==1).sum() / res['nobs']
        res['X']['regiao']['03'] = (self.X[:,[25]]==1).sum() / res['nobs']
        res['X']['regiao']['04'] = (self.X[:,[26]]==1).sum() / res['nobs']
        res['X']['regiao']['05'] = (self.X[:,[27]]==1).sum() / res['nobs']
        res['X']['regiao']['06'] = (self.X[:,[28]]==1).sum() / res['nobs']
        res['X']['regiao']['07'] = (self.X[:,[29]]==1).sum() / res['nobs']
        res['X']['regiao']['08'] = (self.X[:,[30]]==1).sum() / res['nobs']
        res['X']['regiao']['09'] = (self.X[:,[31]]==1).sum() / res['nobs']
        res['X']['regiao']['10'] = (self.X[:,[32]]==1).sum() / res['nobs']
        res['X']['regiao']['11'] = len(self.X[:,23:63][np.where(~self.X[:,23:63].any(axis=1))[0]]) / res['nobs']
        res['X']['regiao']['12'] = (self.X[:,[33]]==1).sum() / res['nobs']
        res['X']['regiao']['13'] = (self.X[:,[34]]==1).sum() / res['nobs']
        res['X']['regiao']['14'] = (self.X[:,[35]]==1).sum() / res['nobs']
        res['X']['regiao']['15'] = (self.X[:,[36]]==1).sum() / res['nobs']
        res['X']['regiao']['16'] = (self.X[:,[37]]==1).sum() / res['nobs']
        res['X']['regiao']['17'] = (self.X[:,[38]]==1).sum() / res['nobs']
        res['X']['regiao']['18'] = (self.X[:,[39]]==1).sum() / res['nobs']
        res['X']['regiao']['19'] = (self.X[:,[40]]==1).sum() / res['nobs']
        res['X']['regiao']['20'] = (self.X[:,[41]]==1).sum() / res['nobs']
        res['X']['regiao']['21'] = (self.X[:,[42]]==1).sum() / res['nobs']
        res['X']['regiao']['22'] = (self.X[:,[43]]==1).sum() / res['nobs']
        res['X']['regiao']['23'] = (self.X[:,[44]]==1).sum() / res['nobs']
        res['X']['regiao']['24'] = (self.X[:,[45]]==1).sum() / res['nobs']
        res['X']['regiao']['25'] = (self.X[:,[46]]==1).sum() / res['nobs']
        res['X']['regiao']['26'] = (self.X[:,[47]]==1).sum() / res['nobs']
        res['X']['regiao']['27'] = (self.X[:,[48]]==1).sum() / res['nobs']
        res['X']['regiao']['28'] = (self.X[:,[49]]==1).sum() / res['nobs']
        res['X']['regiao']['29'] = (self.X[:,[50]]==1).sum() / res['nobs']
        res['X']['regiao']['30'] = (self.X[:,[51]]==1).sum() / res['nobs']
        res['X']['regiao']['31'] = (self.X[:,[52]]==1).sum() / res['nobs']
        res['X']['regiao']['32'] = (self.X[:,[53]]==1).sum() / res['nobs']
        res['X']['regiao']['33'] = (self.X[:,[54]]==1).sum() / res['nobs']
        res['X']['regiao']['34'] = (self.X[:,[55]]==1).sum() / res['nobs']
        res['X']['regiao']['35'] = (self.X[:,[56]]==1).sum() / res['nobs']
        res['X']['regiao']['36'] = (self.X[:,[57]]==1).sum() / res['nobs']
        res['X']['regiao']['37'] = (self.X[:,[58]]==1).sum() / res['nobs']
        res['X']['regiao']['38'] = (self.X[:,[59]]==1).sum() / res['nobs']
        res['X']['regiao']['39'] = (self.X[:,[60]]==1).sum() / res['nobs']
        res['X']['regiao']['40'] = (self.X[:,[61]]==1).sum() / res['nobs']
        res['X']['regiao']['41'] = (self.X[:,[62]]==1).sum() / res['nobs']
        res['X']['sexo'] = {}
        res['X']['sexo']['M'] = len(self.X[:,[63]][self.X[:,[63]]==0]) / res['nobs']
        res['X']['sexo']['F'] = (self.X[:,[63]]==1).sum() / res['nobs']
        res['X']['cod_cont'] = {}
        res['X']['cod_cont']['1'] = len(self.X[:,[65]][self.X[:,[65]]==0]) / res['nobs']
        res['X']['cod_cont']['2'] = (self.X[:,[65]]==1).sum() / res['nobs']
        res['X']['bonus'] = {}
        res['X']['bonus']['0'] = len(self.X[:,66:75][np.where(~self.X[:,66:75].any(axis=1))[0]]) / res['nobs']
        res['X']['bonus']['1'] = (self.X[:,[66]]==1).sum() / res['nobs']
        res['X']['bonus']['2'] = (self.X[:,[67]]==1).sum() / res['nobs']
        res['X']['bonus']['3'] = (self.X[:,[68]]==1).sum() / res['nobs']
        res['X']['bonus']['4'] = (self.X[:,[69]]==1).sum() / res['nobs']
        res['X']['bonus']['5'] = (self.X[:,[70]]==1).sum() / res['nobs']
        res['X']['bonus']['6'] = (self.X[:,[71]]==1).sum() / res['nobs']
        res['X']['bonus']['7'] = (self.X[:,[72]]==1).sum() / res['nobs']
        res['X']['bonus']['8'] = (self.X[:,[73]]==1).sum() / res['nobs']
        res['X']['bonus']['9'] = (self.X[:,[74]]==1).sum() / res['nobs']
        res['X']['tipo_franq'] = {}
        res['X']['tipo_franq']['1'] = (self.X[:,[75]]==1).sum() / res['nobs']
        res['X']['tipo_franq']['2'] = len(self.X[:,75:79][np.where(~self.X[:,75:79].any(axis=1))[0]]) / res['nobs']
        res['X']['tipo_franq']['3'] = (self.X[:,[76]]==1).sum() / res['nobs']
        res['X']['tipo_franq']['4'] = (self.X[:,[77]]==1).sum() / res['nobs']
        res['X']['tipo_franq']['9'] = (self.X[:,[78]]==1).sum() / res['nobs']

        return res
 

class Poisson_model(Data):
    '''
    Provides estimation of Poisson regression model, MLE and PMLE coincide.

    Parameters:
    ----------
    period, aa
    dtype, 2-tuple with values in {'cas', 'rcd', 'app', 'out'} X {'count'}
    grab_results - bypasses optimization procedure, grabing results from file instead
    '''

    def __init__(self, period, aa, dtype, grab_results='no'):
        if grab_results == 'no':
            if dtype[1] != 'count':
                raise Exception('Count data must be provided to Poisson regression model')
    
            super().__init__(period, aa, dtype)
            X = self.X
            y = self.y
    
            def log_likelihood(beta):
                '''Log-likelihood for Poisson regression model'''
            
                res = np.sum(-y * np.dot(X, beta) + np.exp(np.dot(X, beta)) + np.log(factorial(y)))
                return res
            
            def gradient(beta):
                '''Gradient of Log-likelihood for Poisson regression model'''
            
                aux_vec = -y + np.exp(np.dot(X, beta))
                res = (aux_vec[:, np.newaxis] *  X).sum(axis=0)
                return res
    
            x0 = np.zeros(len(X[0]))
            x0[0] = 1
            x0[1] = np.log(sum(y)/len(y))
            prec_param = 1e-4
            bounds = ((1 - prec_param, 1 + prec_param),)
            for i in range(len(X[0])-1):
                bounds += ((None, None),)
            
            res = minimize(log_likelihood, x0, method='TNC', jac=gradient, bounds=bounds, options={'disp': True})
            if res.success == 0:
                res = minimize(log_likelihood, x0, method='L-BFGS-B', jac=gradient, bounds=bounds, options={'disp': True})
                if res.success == 0:
                    res = minimize(log_likelihood, x0, method='SLSQP', jac=gradient, bounds=bounds, options={'disp': True})
    
            self.fit_success = res.success
            self.fit_x = res.x
            self.fit_fun = res.fun
        
        elif grab_results == 'yes':
            if dtype[1] != 'count':
                raise Exception('Count data must be provided to Poisson regression model')
            
            if not os.path.exists('/home/pgsqldata/Susep/PoissonResults_' + dtype[0] + '.db'):
                raise Exception('File PoissonResults_' + dtype[0] + '.db not found')

            super().__init__(period, aa, dtype)
            x_res_dict = shelve.open('/home/pgsqldata/Susep/PoissonResults_' + dtype[0] + '.db')
            self.fit_success = x_res_dict[period+aa]['success']
            self.fit_x = np.insert(x_res_dict[period+aa]['coeffs'], 0, 1)
            self.fit_fun = x_res_dict[period+aa]['-ln L']
            x_res_dict.close()

        else:
            print('Invalid input for grab_results parameter: must be either "yes" or "no"')

    def var_MLH(self):
        '''
        Variance for Poisson MLE using Hessian
        For all variances, nan's are inserted where beta=0
        '''
    
        mu = np.exp(np.dot(self.X, self.fit_x))[:, np.newaxis]
        index0 = np.where(self.fit_x[1:] == 0)[0]
        X = np.delete(self.X[:, 1:], index0, 1)
        var = np.linalg.inv((X * mu).T @ X)
        std = np.sqrt(np.diag(var))
        var = np.insert(var, index0, np.nan, axis=0)
        var = np.insert(var, index0, np.nan, axis=1)
        std = np.insert(std, index0, np.nan)
        return (var, std)

    def var_MLOP(self):
        '''Variance for Poisson MLE using summed outer product of first derivatives'''
    
        mu = np.exp(np.dot(self.X, self.fit_x))[:, np.newaxis]
        index0 = np.where(self.fit_x[1:] == 0)[0]
        X = np.delete(self.X[:, 1:], index0, 1)
        y = self.y[:, np.newaxis]
        var = np.linalg.inv((X * np.square(y - mu)).T @ X)
        std = np.sqrt(np.diag(var))
        var = np.insert(var, index0, np.nan, axis=0)
        var = np.insert(var, index0, np.nan, axis=1)
        std = np.insert(std, index0, np.nan)
        return (var, std)

    def var_NB1(self):
        '''
        Variance assuming var = phi * mu
        Returns parameter phi
        '''

        mu = np.exp(np.dot(self.X, self.fit_x))[:, np.newaxis]
        index0 = np.where(self.fit_x[1:] == 0)[0]
        X = np.delete(self.X[:, 1:], index0, 1)
        y = self.y[:, np.newaxis]
        phi = (len(X) - np.shape(self.X[:, 1:])[1])**(-1) * (np.square(y - mu)/mu).sum()
        var = phi * np.linalg.inv((X * mu).T @ X)
        std = np.sqrt(np.diag(var))
        var = np.insert(var, index0, np.nan, axis=0)
        var = np.insert(var, index0, np.nan, axis=1)
        std = np.insert(std, index0, np.nan)
        return (var, std, phi)

    def var_NB2(self):
        '''
        Variance assuming var = mu + alpha * mu^2
        Returns parameter alpha
        '''

        mu = np.exp(np.dot(self.X, self.fit_x))[:, np.newaxis]
        index0 = np.where(self.fit_x[1:] == 0)[0]
        X = np.delete(self.X[:, 1:], index0, 1)
        y = self.y[:, np.newaxis]
        alpha = (len(X) - np.shape(self.X[:, 1:])[1])**(-1) * ((np.square(y - mu) - mu) / np.square(mu)).sum()
        var = np.linalg.inv((X * mu).T @ X) @ ((X * (mu + alpha * np.square(mu))).T @ X) @ np.linalg.inv((X * mu).T @ X) 
        std = np.sqrt(np.diag(var))
        var = np.insert(var, index0, np.nan, axis=0)
        var = np.insert(var, index0, np.nan, axis=1)
        std = np.insert(std, index0, np.nan)
        return (var, std, alpha)

    def var_RS(self):
        '''
        Variance estimator robust to specification error
        '''

        mu = np.exp(np.dot(self.X, self.fit_x))[:, np.newaxis]
        index0 = np.where(self.fit_x[1:] == 0)[0]
        X = np.delete(self.X[:, 1:], index0, 1)
        y = self.y[:, np.newaxis]
        var = np.linalg.inv((X * mu).T @ X) @ ((X * (np.square(y - mu))).T @ X) @ np.linalg.inv((X * mu).T @ X) 
        std = np.sqrt(np.diag(var))
        var = np.insert(var, index0, np.nan, axis=0)
        var = np.insert(var, index0, np.nan, axis=1)
        std = np.insert(std, index0, np.nan)
        return (var, std)

    def save_results(self, period, aa, dtype):
        if self.fit_success == 0:
            with open('models_Poisson.log', 'a') as log_file:
                log_file.write('\n' + period + aa + dtype[0] + ' coeffs failed')
        else:
            try:
                x_res_dict = {'desc_stats': self.desc_stats(), 'success': self.fit_success, '-ln L': self.fit_fun, 'coeffs': self.fit_x[1:], 'var_MLH': self.var_MLH()[0], 'std_MLH': self.var_MLH()[1], 'var_MLOP': self.var_MLOP()[0], 'std_MLOP': self.var_MLOP()[1], 'var_NB1': self.var_NB1()[0], 'std_NB1': self.var_NB1()[1], 'phi_NB1': self.var_NB1()[2], 'var_RS': self.var_RS()[0], 'std_RS': self.var_RS()[1]}
            except:
                x_res_dict = {'desc_stats': self.desc_stats(), 'success': self.fit_success, '-ln L': self.fit_fun, 'coeffs': self.fit_x[1:]}
                try:
                    x_res_dict['var_MLH'] =  self.var_MLH()[0]
                    x_res_dict['std_MLH'] =  self.var_MLH()[1]
                except:
                    x_res_dict['var_MLH'] =  np.zeros((len(x_res_dict['coeffs']), len(x_res_dict['coeffs'])))
                    x_res_dict['std_MLH'] =  np.zeros(len(x_res_dict['coeffs']))
                    with open('models_Poisson.log', 'a') as log_file:
                        log_file.write('\n' + period + aa + dtype[0] + ' var_MLH failed')
                try:
                    x_res_dict['var_MLOP'] =  self.var_MLOP()[0]
                    x_res_dict['std_MLOP'] =  self.var_MLOP()[1]
                except:
                    x_res_dict['var_MLOP'] =  np.zeros((len(x_res_dict['coeffs']), len(x_res_dict['coeffs'])))
                    x_res_dict['std_MLOP'] =  np.zeros(len(x_res_dict['coeffs']))
                    with open('models_Poisson.log', 'a') as log_file:
                        log_file.write('\n' + period + aa + dtype[0] + ' var_MLOP failed')
                try:
                    x_res_dict['var_NB1'] =  self.var_NB1()[0]
                    x_res_dict['std_NB1'] =  self.var_NB1()[1]
                    x_res_dict['phi_NB1'] =  self.var_NB1()[2]
                except:
                    x_res_dict['var_NB1'] =  np.zeros((len(x_res_dict['coeffs']), len(x_res_dict['coeffs'])))
                    x_res_dict['std_NB1'] =  np.zeros(len(x_res_dict['coeffs']))
                    x_res_dict['phi_NB1'] =  0
                    with open('models_Poisson.log', 'a') as log_file:
                        log_file.write('\n' + period + aa + dtype[0] + ' var_NB1 failed')
                try:
                    x_res_dict['var_RS'] =  self.var_RS()[0]
                    x_res_dict['std_RS'] =  self.var_RS()[1]
                except:
                    x_res_dict['var_RS'] =  np.zeros((len(x_res_dict['coeffs']), len(x_res_dict['coeffs'])))
                    x_res_dict['std_RS'] = np.zeros(len(x_res_dict['coeffs']))
                    with open('models_Poisson.log', 'a') as log_file:
                        log_file.write('\n' + period + aa + dtype[0] + ' var_RS failed')
    
            db_file = '/home/pgsqldata/Susep/PoissonResults_' + dtype[0] + '.db'
            db = shelve.open(db_file)
            db[period+aa] = x_res_dict
            db.close()
            print('Results from Poisson class instance for period ' + period + aa + ' of type ' + dtype[0] + ' saved in db file')


class Poisson_tests(Data):
    '''
    Provides methods for testing adequacy of models

    Parameters:
    ----------
    dtype, 2-tuple with values in {'cas', 'rcd', 'app', 'out'} X {'count'}
    '''

    def __init__(self, model, period, aa, dtype):
        self.model = model
        dbfile = '/home/pgsqldata/Susep/' + model + 'Results_' + dtype[0] + '.db'
        if not os.path.exists(dbfile): 
            raise Exception('File ' + dbfile + ' not found')

        super().__init__(period, aa, dtype)
        x_res_dict = shelve.open(dbfile)
        self.fit_success = x_res_dict[period+aa]['success']
        if self.fit_success == 0:
            raise Exception('Model optimization failed to converge for ' + model + period + aa)

        self.fit_x = np.insert(x_res_dict[period+aa]['coeffs'], 0, 1)
        self.fit_fun = x_res_dict[period+aa]['-ln L']
        self.desc_stats = x_res_dict[period+aa]['desc_stats']
        if model in {'Poisson'}:
            self.var_MLH = x_res_dict[period+aa]['var_MLH']
            self.std_MLH = x_res_dict[period+aa]['std_MLH']
            self.var_MLOP = x_res_dict[period+aa]['var_MLOP']
            self.std_MLOP = x_res_dict[period+aa]['std_MLOP']
            self.var_NB1 = x_res_dict[period+aa]['var_NB1']
            self.std_NB1 = x_res_dict[period+aa]['std_NB1']
            self.phi_NB1 = x_res_dict[period+aa]['phi_NB1']
            self.var_RS = x_res_dict[period+aa]['var_RS']
            self.std_RS = x_res_dict[period+aa]['std_RS']
        
        x_res_dict.close()
        self.mu = np.exp(np.dot(self.X, self.fit_x))
    
    def fitted_freqs(self):
        '''
        Fitted frequencies according to model specification
        
        ##Obtain Chi-Square distribuion of statistic in Andrews(88)
        '''

        if self.model == 'Poisson':
            p_j = {}
            for j in range(np.unique(self.y).max()):
                p_j[str(j)] = (np.multiply(np.exp(-self.mu), self.mu**j) / factorial(j)).sum() / len(self.X)
        else:
            pass

        return p_j

    def pearson_stat(self):
        '''
        Pearson statistic equals sum_i[(y_i - mu_i)^2 / omega_i]
        If model specification is Poisson, omega_i = mu_i
        
        ## Obtain Chi-Square distribuion of statistic in McCullagh(86)
        '''

        if self.model == 'Poisson':
            ps = self.phi_NB1 * (np.shape(self.X)[0] - np.shape(self.X[:, 1:])[1])
            n_minus_k = (np.shape(self.X)[0] - np.shape(self.X[:, 1:])[1])
        else:
            pass

        return (ps, n_minus_k)

    def deviance_r2(self):
        '''
        Pseudo R^2 according to 1 - (D(y, mu)/D(y, y_barr) or 1 - RSS/TSS
        R^2 for Poisson: sum[y_i * ln(mu_i/y_bar) - (y_i - mu_i)] / sum [y_i * ln(y_i/y_bar)] 
        '''

        if self.model == 'Poisson':
            r2_dev = (self.y * np.log(self.mu / np.mean(self.y)) - (self.y - self.mu)).sum() / np.where(self.y == 0, 0, self.y * np.log(self.y / np.mean(self.y))).sum()
        else:
            pass

        return r2_dev

    def residuals(self, submodel='standard'):
        '''
        Computes Pearson (p), deviance (d) and Anscombe (a) residuals
        For Pearson residuals, Poisson NB1 variance phi * mu is used 
        '''

        if self.model == 'Poisson' and submodel == 'standard':
            p = (self.y - self.mu) / np.sqrt(self.mu)
        elif self.model == 'Poisson' and submodel == 'NB1':
            p = (self.y - self.mu) / np.sqrt(self.phi_NB1 * self.mu)
        else:
            pass

        if self.model == 'Poisson':
            d = np.where(self.y - self.mu >= 0, 1, -1) * np.sqrt(2 * (np.where(self.y ==0, 0, self.y * np.log(self.y / np.mean(self.y))) - (self.y - self.mu)))
            a = 1.5 * (self.y**(2/3) - self.mu**(2/3)) / self.mu**(1/6)

        return (p, d, a)

class Binary_models(Data):
    '''
    Provides estimation of binary outcome regression models.
    Distributions used: logistic, normal and poisson

    Parameters:
    ----------
    distribution, with possible values in {'Logistic', 'Normal', 'Poisson'}
    period, aa
    dtype, 2-tuple with values in {'cas', 'rcd', 'app', 'out'} X {'count'}
    grab_results - bypasses optimization procedure, grabing results from file instead
    '''

    def __init__(self, distribution, period, aa, dtype, grab_results='no'):
        self.distribution = distribution
        self.period = period
        self.aa = aa
        self.dtype = dtype
        super().__init__(period, aa, dtype, binary_count='yes')
        if grab_results == 'no':
            X = self.X
            y = self.y
            self.desc_stats = self.desc_stats()
            desc_stats = self.desc_stats
            if distribution == 'Logit': 
                def log_likelihood(beta):
                    '''Log-likelihood for Binary Logit model'''
                
                    res = -1 * (np.sum(y * np.log(np.exp(np.dot(X, beta)) / (1 + np.exp(np.dot(X, beta)))) + (1 - y) * np.log(1 - np.exp(np.dot(X, beta)) / (1 + np.exp(np.dot(X, beta))))))
                    return res
                
                def gradient(beta):
                    '''Gradient of Log-likelihood for Binary Logit model'''
                
                    aux_vec = y - np.exp(np.dot(X, beta)) / (1 + np.exp(np.dot(X, beta)))
                    res = -1 * (aux_vec[:, np.newaxis] *  X).sum(axis=0)
                    return res

                def var_aux_ML(X, beta):
                    '''
                    Variance term before inversion for binary outcome model using logistic distribution:
                    [sum_i exp(x_i'beta)/(1+exp(x_i'beta))^2 * x_i * x_i']^(-1) 
                    '''
                
                    lambda_prime = (np.exp(np.dot(X, beta))/(1 + np.exp(np.dot(X, beta)))**2)[:, np.newaxis]
                    index0 = np.where(beta[1:] == 0)[0]
                    X = np.delete(X[:, 1:], index0, 1)
                    var_aux = (X * lambda_prime).T @ X
                    return (var_aux, index0)
                    
            
            elif distribution == 'Probit': 
                def log_likelihood(beta):
                    '''Log-likelihood for Binary Probit model'''
                
                    res = -1 * (np.sum(y * np.log(norm.cdf(np.dot(X, beta))) + (1 - y) * np.log(1 - norm.cdf(np.dot(X, beta)))))
                    return res
                
                def gradient(beta):
                    '''Gradient of Log-likelihood for Binary Probit model'''
                
                    aux_vec = norm.pdf(np.dot(X, beta))/(norm.cdf(np.dot(X, beta)) * (1 - norm.cdf(np.dot(X, beta)))) * (y - norm.cdf(np.dot(X, beta)))
                    res = -1 * (aux_vec[:, np.newaxis] *  X).sum(axis=0)
                    return res

                def var_aux_ML(X, beta):
                    '''
                    Variance term before inversion for binary outcome model using normal distribution:
                    [sum_i phi(x_i'beta)^2/Phi(x_i'beta)(1-Phi(x_i'beta)) * x_i * x_i']^(-1) 
                    where Phi and phi are the cdf and pdf of the normal distribution
                    '''
                
                    weight = (norm.pdf(np.dot(X, beta))**2/(norm.cdf(np.dot(X, beta)) * (1 - norm.cdf(np.dot(X, beta)))))[:, np.newaxis]
                    index0 = np.where(beta[1:] == 0)[0]
                    X = np.delete(X[:, 1:], index0, 1)
                    var_aux = (X * weight).T @ X
                    return (var_aux, index0)

            elif distribution == 'Poisson': 
                def log_likelihood(beta):
                    '''
                    Log-likelihood for Binary Poisson model, where distribution for Bernoulli parameter p is specified as:
                    F = 1 - exp(-exp(x_i'beta)), the probability of y > 0 for the Poisson distribution, and mu = exp(x_i'beta)
                    Due to proximity of mu_i to zero, a lower bound equal to min np.log such that np.log > -inf is imposed
                    '''
                
                    F0 = np.exp(-np.exp(np.dot(X, beta)))
                    F1 = 1 - np.exp(-np.exp(np.dot(X, beta)))
                    F1[F1 < lb_log] = lb_log
                    res = -1 * np.sum(y * np.log(F1) + (1 - y) * np.log(F0))
                    return res

                def gradient(beta):
                    '''
                    Gradient of Log-likelihood for Binary Poisson model
                    Formula is obtained combining Cameron, Trivedi (05) eq. 14.5 and F = 1 - exp(-exp(x_i'beta)), F' = beta_k * exp(-exp(x_i'beta)+x_i'beta):
                    sum_i (y_i - 1 + exp(-exp(X_i'beta))/[exp(-exp(x_i'beta)) - exp(-2exp(x_i'beta))] * exp(-exp(x_i'beta)+x_i'beta) * x_i * beta_k
                    Denominator is restricted to system limit min to avoid inf ratio
                    '''
                
                    F0 = np.exp(-np.exp(np.dot(X, beta)))
                    F1 = 1 - np.exp(-np.exp(np.dot(X, beta)))
                    denominator = np.exp(-np.exp(np.dot(X, beta))) - np.exp(-2*np.exp(np.dot(X, beta)))
                    denominator[denominator < lb_ratio] = lb_ratio
                    F1_prime = np.exp(-np.exp(np.dot(X, beta)) + np.dot(X, beta))
                    aux_vec = ((y - F1) / denominator) * F1_prime
                    res = -1 * (aux_vec[:, np.newaxis] *  X).sum(axis=0)
                    return res

                def var_aux_ML(X, beta):
                    '''
                    Variance term before inversion for binary outcome model using poisson distribution
                    Obtained by substituting F = 1 - exp(-exp(x_i'beta)) in eq. 14.7, Cameron and Trivedi (05)
                    Variance for binary outcomes has simple form (sum_i weight * x_i * x_i')^(-1) where weight = F'^2/F1*F0
                    Denominator of weights may be zero if F1 is sufficiently close to zero given mu_i - this is a rare event and the solution given here consists of making the weight equal to zero for such i
                    '''

                    denominator = np.exp(-np.exp(np.dot(X, beta))) - np.exp(-2*np.exp(np.dot(X, beta)))
                    F1_prime = np.exp(-np.exp(np.dot(X, beta)) + np.dot(X, beta))
                    index1 = np.where(denominator < lb_ratio)
                    denominator[index1] = lb_ratio
                    F1_prime[index1] = 0.
                    weight = (np.square(F1_prime) / denominator)[:, np.newaxis]
                    index0 = np.where(beta[1:] == 0)[0]
                    X = np.delete(X[:, 1:], index0, 1)
                    var_aux = (X * weight).T @ X
                    return (var_aux, index0)
    
            def save_results(coeffs, var, std, desc_stats, distribution, period, aa, dtype):
                if coeffs.success == 0:
                    with open('models_BinaryResults.log', 'a') as log_file:
                        log_file.write('\n' + distribution + period + aa + dtype[0] + ' coeffs failed')
                else:
                    x_res_dict = {'desc_stats': desc_stats, '-ln L': coeffs.fun, 'coeffs': coeffs.x[1:], 'var_ML': var, 'std_ML': std}
                    db_file = '/home/pgsqldata/Susep/BinaryModelResults_' + distribution + '_' + dtype[0] + '.db'
                    db = shelve.open(db_file)
                    db[period+aa] = x_res_dict
                    db.close()
                    print('Results from Binary Model class instance, distribution ' + distribution + ', for period ' + period + aa + ' of type ' + dtype[0] + ' saved in db file')

            x0 = np.zeros(len(X[0]))
            x0[0] = 1
            x0[1] = np.log(sum(y)/len(y))
            bounds = ((1 - prec_param, 1 + prec_param),)
            for i in range(len(X[0])-1):
                bounds += ((None, None),)
            
            coeffs = minimize(log_likelihood, x0, method='TNC', jac=gradient, bounds=bounds, options={'disp': True})
            if coeffs.success == 0:
                coeffs = minimize(log_likelihood, x0, method='L-BFGS-B', jac=gradient, bounds=bounds, options={'disp': True})
                if coeffs.success == 0:
                    coeffs = minimize(log_likelihood, x0, method='SLSQP', jac=gradient, bounds=bounds, options={'disp': True})
            
            var_aux, index0 = var_aux_ML(X, coeffs.x)
            try:
                var = np.linalg.inv(var_aux)
            except:
                _, index2 = sympy.Matrix(var_aux).rref()
                var_aux = var_aux[index2, :][:, index2]
                var = np.linalg.inv(var_aux)
                index3 = list(set(range(len(var_aux))) - set(index2))
                var = np.insert(var, index3, np.nan, axis=0)
                var = np.insert(var, index3, np.nan, axis=1)
                with open('models_BinaryResults.log', 'a') as log_file:
                    log_file.write('\n' + distribution + period + aa + dtype[0] + ' var_aux_ML singular, removed dependent columns: ', index3)

            var = np.insert(var, index0, np.nan, axis=0)
            var = np.insert(var, index0, np.nan, axis=1)
            std = np.sqrt(np.diag(var))
            save_results(coeffs, var, std, desc_stats, distribution, period, aa, dtype)
            self.success = coeffs.success
            self.x = coeffs.x
            self.fun = coeffs.fun
            self.var_ML = var
            self.std_ML = std

        elif grab_results == 'yes':
            x_res_dict = shelve.open('/home/pgsqldata/Susep/BinaryModelResults_' + distribution + '_' + dtype[0] + '.db')
            self.desc_stats = x_res_dict[period+aa]['desc_stats']
            self.x = np.insert(x_res_dict[period+aa]['coeffs'], 0, 1)
            self.fun = x_res_dict[period+aa]['-ln L']
            self.var_ML = x_res_dict[period+aa]['var_ML']
            self.std_ML = x_res_dict[period+aa]['std_ML']
            x_res_dict.close()

    def pseudo_R2(self):
        '''
        Relative gains pseudo-R2 as proposed by McFadden(74)
        '''

        if self.distribution == 'Logit':
            p_hat = np.exp(np.dot(self.X, self.x)) / (1 + np.exp(np.dot(self.X, self.x)))
        elif self.distribution == 'Probit':
            p_hat = norm.cdf(np.dot(self.X, self.x))
        elif self.distribution == 'Poisson':
            p_hat = 1 - np.exp(-np.exp(np.dot(self.X, self.x)))

        N = len(self.y)
        y_bar = np.mean(self.y)
        index0 = np.where(p_hat < lb_log)
        p_hat[index0] = lb_log
        R2 = 1 - (self.y * np.log(p_hat) + (1 - self.y) * np.log(1 - p_hat)).sum() / (N * (y_bar * np.log(y_bar) + (1 - y_bar) * np.log(1 - y_bar)))
        db_file = '/home/pgsqldata/Susep/BinaryModelResults_' + self.distribution + '_' + self.dtype[0] + '.db'
        db = shelve.open(db_file)
        db[self.period+self.aa]['pseudo_R2'] = R2
        db.close()

        return R2


if __name__ == '__main__':
#    periods = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')
    periods = ('1tr', '2tr', '3tr', '4tr')
#    dtypes =(('cas', 'count'), ('rcd', 'count'), ('app', 'count'))
    dtypes =(('cas', 'count'), ('rcd', 'count'))
    years = ('08', '09', '10', '11')
    distributions = ('Poisson', 'Logit', 'Probit')
    for period in periods:
        for aa in years:
            for dtype in dtypes:
                for distribution in distributions:
                    if period == '1tr' and distribution != 'Poisson':
                        continue
                    if period == '1tr' and aa == '08' and distribution == 'Poisson':
                        continue
                    if period == '1tr' and aa == '09' and dtype[0] == 'rcd' and distribution == 'Poisson':
                        continue
                    if period == '1tr' and aa == '10' and distribution == 'Poisson':
                        continue
                    if period == '2tr' and aa == '08' and distribution != 'Poisson':
                        continue
                    print('Next regression: ' + distribution + '_' + period + aa + dtype[0])
                    Binary_models(distribution, period, aa, dtype)
