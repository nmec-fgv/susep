#########################################################################
## Regression models of counts and claims data for auto insurance cost ##
#########################################################################


import os
import pickle
import numpy as np
import shelve
from scipy.special import factorial
from scipy.optimize import minimize
import pdb


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
    Data preparation for subsequently running model.
    Loads data from files 'data_mmmaa.pkl' according to period request.
    Returns X', and 'y' according to data type requested ({'cas', 'rcd', 'app', 'out'} X {'claim', 'count'}).
    
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
        X[:,[64]] = X[:,[64]] / 100
        X[:,79:83] = X[:,79:83] / 100000
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
 

class Poisson(Data):
    '''
    Provides estimation of Poisson regression model, MLE and PMLE coincide.

    Parameters:
    ----------
    data, must be data attribute previously generated from data class call
    w
    dtype, 2-tuple with values in {'cas', 'rcd', 'app', 'out'} X {'count'}
    '''

    def __init__(self, period, aa, dtype):
        if dtype[1] != 'count':
            raise Exception('Count data must be provided to Poisson regression model')

        super().__init__(period, aa, dtype)

        def log_likelihood(beta):
            '''Log-likelihood of Poisson regression model'''
        
            res = np.sum(-y * np.dot(X, beta) + np.exp(np.dot(X, beta)) + np.log(factorial(y)))
            return res
        
        def gradient(beta):
            '''Gradient of Log-likelihood of Poisson regression model'''
        
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

        self.fit = res


    def var_MLH(self):
        '''
        Variance for Poisson MLE using Hessian
        For all variances, nan's are inserted where beta=0
        '''
    
        index0 = np.where(self.fit.x == 0)[0]
        X = np.delete(self.X, index0, 1)
        beta = np.delete(self.fit.x, index0)
        mu = np.exp(np.dot(X, beta))[:, np.newaxis]
        var = np.linalg.inv((X * mu).T @ X)
        std = np.sqrt(np.diag(var))
        var = np.insert(var, index0, np.nan, axis=0)
        var = np.insert(var, index0, np.nan, axis=1)
        std = np.insert(std, index0, np.nan)
        return (var, std)

    def var_MLOP(self):
        '''Variance for Poisson MLE using summed outer product of first derivatives'''
    
        index0 = np.where(self.fit.x == 0)[0]
        X = np.delete(self.X, index0, 1)
        y = self.y[:, np.newaxis]
        beta = np.delete(self.fit.x, index0)
        mu = np.exp(np.dot(X, beta))[:, np.newaxis]
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

        index0 = np.where(self.fit.x == 0)[0]
        X = np.delete(self.X, index0, 1)
        y = self.y[:, np.newaxis]
        beta = np.delete(self.fit.x, index0)
        mu = np.exp(np.dot(X, beta))[:, np.newaxis]
        phi = (len(X) - np.shape(self.X)[1])**(-1) * (np.square(y - mu)/mu).sum()
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

        index0 = np.where(self.fit.x == 0)[0]
        X = np.delete(self.X, index0, 1)
        y = self.y[:, np.newaxis]
        beta = np.delete(self.fit.x, index0)
        mu = np.exp(np.dot(X, beta))[:, np.newaxis]
        alpha = (len(X) - np.shape(self.X)[1])**(-1) * ((np.square(y - mu) - mu) / np.square(mu)).sum()
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

        index0 = np.where(self.fit.x == 0)[0]
        X = np.delete(self.X, index0, 1)
        y = self.y[:, np.newaxis]
        beta = np.delete(self.fit.x, index0)
        mu = np.exp(np.dot(X, beta))[:, np.newaxis]
        var = np.linalg.inv((X * mu).T @ X) @ ((X * (np.square(y - mu))).T @ X) @ np.linalg.inv((X * mu).T @ X) 
        std = np.sqrt(np.diag(var))
        var = np.insert(var, index0, np.nan, axis=0)
        var = np.insert(var, index0, np.nan, axis=1)
        std = np.insert(std, index0, np.nan)
        return (var, std)


if __name__ == '__main__':
#    periods = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez', '1tr', '2tr', '3tr', '4tr')
    periods = ('1tr', '2tr', '3tr', '4tr')
    years = ('08', '09', '10', '11')
    dtypes =(('cas', 'count'), ('rcd', 'count'), ('app', 'count'))
    for period in periods:
        for aa in years:
            for dtype in dtypes:
                db_file = '/home/pgsqldata/Susep/PoissonResults_' + dtype[0] + '.db'
                x = Poisson(period, aa, dtype)
                x_res_dict = {'-ln L': x.fit.fun, 'coeffs': x.fit.x, 'var_MLH': x.var_MLH()[0], 'std_MLH': x.var_MLH()[1], 'var_MLOP': x.var_MLOP()[0], 'std_MLOP': x.var_MLOP()[1], 'var_NB1': x.var_NB1()[0], 'std_NB1': x.var_NB1()[1], 'phi_NB1': x.var_NB1()[2], 'var_RS': x.var_RS()[0], 'std_RS': x.var_RS()[1]}
                db = shelve.open(db_file)
                db[period+aa] = x_res_dict
                db.close()
                print('Instance from Poisson class for period ' + period + aa + ' of type ' + dtype[0] + ' made persistent in db file')
