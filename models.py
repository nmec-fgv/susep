#########################################################################
## Regression models of counts and claims data for auto insurance cost ##
#########################################################################


import os
import sys
import pickle
import shelve
import numpy as np
import sympy
import pdb
import time


# Lower bound parameters for logs and ratios, plus precision parameter:

lb_log = 1e-323
lb_ratio = 1e-308
lb_alpha = 1e-77
prec_param = 1e-4


# Data directory:

data_dir = 'persistent/'


# Auxiliary functions:

def file_load(filename):
    try:
        os.path.exists(data_dir + filename)
        with open(data_dir + filename, 'rb') as file:
            x = pickle.load(file)
    except:
        print('File ' + filename + ' not found')

    return x

def save_results(res_dict, model, claim_type):
    db_file = data_dir + 'results_' + claim_type + '.db'
    db = shelve.open(db_file)
    db[model] = res_dict
    db.close()
    print('Results from ' + model + ' model, for claim_type ' + claim_type + ', saved in db file')
    return

def grab_results(model, claim_type, keys=None):
    db_file = data_dir + 'results_' + claim_type + '.db'
    if not os.path.exists(db_file): 
        raise Exception('File ' + dbfile + ' not found')

    db = shelve.open(db_file)
    if keys == None:
        res = db[model]
    else:
        res = {}
        for key in keys:
            res[key] = db[model][key]

    db.close()
    return res
    

# Classes:

class Estimation:
    '''
    Provides estimated parameters of Poisson regression model.

    Parameters:
    -----------
    claim_type - type of claim to be analyzed
    model - specification of density of random component, currently Poisson or Gamma

    Methods:
    --------
    save_results
    '''
    def __init__(self, model, claim_type):
        if claim_type not in {'casco', 'rcd'} or model not in {'Poisson', 'NB2', 'Gamma', 'InvGaussian'}:
            raise Exception('Model or claim_type provided not in permissible set')

        if model in {'Poisson', 'NB2'}:
            dependent = 'freq'
        elif model in {'Gamma', 'InvGaussian'}:
            dependent = 'sev'

        X = file_load(dependent + '_' + claim_type + '_matrix.pkl')
        if model == 'Poisson':
            def LL_func(beta):
                '''
                Log-likelihood for Poisson regression model, constant term excluded
                -exposure_i * exp(x_i'beta) + y_i * x_i'beta
                '''
            
                res = np.sum(- X[:, [1]] * np.exp(X[:, 2:] @ beta) + X[:, [0]] * X[:, 2:] @ beta)
                return res
            
            def grad_func(beta):
                '''
                Gradient of Log-likelihood for Poisson regression model
                x_i * [y_i - exposure_i * exp(x_i'beta)]
                '''
            
                aux_vec = X[:, [0]] - X[:, [1]] * np.exp(X[:, 2:] @ beta)
                res = (aux_vec.T @ X[:, 2:]).T
                return res
    
            def hess_ninv(beta):
                '''
                Inverse of negative Hessian of Poisson loglikelihood
                mu_i * x_i * x_i'
                '''
    
                res = np.linalg.inv(X[:, 2:].T @ (X[:, [1]] * np.exp(X[:, 2:] @ beta) * X[:, 2:]))
                return res

        elif model == 'NB2':
            def LL_func(beta):
                '''
                Log-likelihood for NB2 regression model, constant term excluded
                '''
            
                res = 0 #### write function
                return res
            
            def grad_func(beta):
                '''
                Gradient of Log-likelihood for NB2 regression model
                ((y_i - mu_i)/(1+alpha*mu_i)] * x_i
                (1/alpha^2)*(ln(1+alpha*mu_i)-sum_{j=0}^{y_i-1}1/(j+alpha^-1)+(y_i-mu_i)/(alpha*(1+alpha*mu_i)
                '''
            
                beta[-1] = np.maximum(beta[-1], lb_alpha)
                mu = X[:, [1]] * np.exp(X[:, 2:] @ beta[:-1])
                aux_vec = (X[:, [0]] - mu) / (1 + beta[-1] * mu)
                aux_beta = (aux_vec.T @ X[:, 2:]).T
                aux_jsum = np.array([np.sum((np.arange(X[i, [0]])+beta[-1]**(-1))**(-1)) for i in range(len(X))])[:, np.newaxis]
                aux_alpha = np.sum(beta[-1]**(-2) * (np.log(1 + beta[-1] * mu) - aux_jsum) + (X[:, [0]] - mu) / (beta[-1] * (1 + beta[-1] * mu))) 
                res = np.vstack((aux_beta, aux_alpha))
                return res
    
            def hess_ninv(beta):
                '''
                Inverse of negative Hessian of NB2 loglikelihood
                mu_i * x_i * x_i'
                '''
    
                beta[-1] = np.maximum(beta[-1], lb_alpha)
                mu = X[:, [1]] * np.exp(X[:, 2:] @ beta[:-1])
                aux_beta = np.linalg.inv(X[:, 2:].T @ ((mu / (1 + beta[-1] * mu)) * X[:, 2:]))
                aux_jsum = np.array([np.sum((np.arange(X[i, [0]])+beta[-1]**(-1))**(-1)) for i in range(len(X))])[:, np.newaxis]
                aux_alpha = (np.sum(beta[-1]**(-4) * (np.log(1 + beta[-1] * mu) - aux_jsum)**2 + mu / (beta[-1]**2 * (1 + beta[-1] * mu))))**(-1)
                res = np.hstack((aux_beta, np.zeros(np.shape(aux_beta)[0])[:, np.newaxis]))
                res = np.vstack((res, np.concatenate((np.zeros(np.shape(res)[1]-1), [aux_alpha]))))
                return res

        elif model == 'Gamma':
            def LL_func(beta):
                '''
                Log-likelihood for Gamma regression model, nu excluded, constant term excluded
                -n_i * x_i'beta -sum c_ik * exp(-x_i'beta) 
                '''
            
                res = np.sum(- X[:, [1]] * X[:, 2:] @ beta - X[:, [0]] * np.exp(X[:, 2:] @ beta))
                return res
            
            def grad_func(beta):
                '''
                Gradient of Log-likelihood for Gamma regression model
                x_i * [y_i - exposure_i * exp(x_i'beta)]
                '''
            
                aux_vec = X[:, [0]] * np.exp(-1 * X[:, 2:] @ beta) - X[:, [1]] 
                res = (aux_vec.T @ X[:, 2:]).T
                return res
    
            def hess_ninv(beta):
                '''
                Inverse of negative Hessian of Gamma loglikelihood
                (sum c_ik / mu_i) * x_i * x_i'
                '''
    
                res = np.linalg.inv(X[:, 2:].T @ (X[:, [0]] * np.exp(-1 * X[:, 2:] @ beta) * X[:, 2:]))
                return res

        elif model == 'InvGaussian':
            def LL_func(beta):
                '''
                Log-likelihood for InvGaussian regression model, sigma excluded, constant term excluded
                -(sum c_ik/2) * exp(-2*x_i'beta) + n_i * exp(-x_i'beta) 
                '''
            
                res = np.sum(- (X[:, [0]] / 2) * np.exp(-2 * X[:, 2:] @ beta) + X[:, [1]] * np.exp(-1 * X[:, 2:] @ beta))
                return res
            
            def grad_func(beta):
                '''
                Gradient of Log-likelihood for InvGaussian regression model
                x_i * [sum c_ik * exp(-2*x_i'beta) - n_i * exp(-x_i'beta)]
                '''
            
                aux_vec = X[:, [0]] * np.exp(-2 * X[:, 2:] @ beta) - X[:, [1]] * np.exp(-1 * X[:, 2:] @ beta)
                res = (aux_vec.T @ X[:, 2:]).T
                return res
    
            def hess_ninv(beta):
                '''
                Inverse of negative Hessian of InvGaussian loglikelihood
                [2 * sum c_ik * exp(-2*x_i'beta) - n_i * exp(-x_i'beta)] * x_i * x_i'
                '''
    
                res = np.linalg.inv(X[:, 2:].T @ ((2 * X[:, [0]] * np.exp(-2 * X[:, 2:] @ beta) - X[:, [1]] * np.exp(-1 * X[:, 2:] @ beta)) * X[:, 2:]))
                return res

        # Initial guesses and stoping parameter:
        beta = np.zeros(np.shape(X)[1] - 2)[:, np.newaxis]
        if model == 'NB2':
            beta = np.vstack((beta, np.array([.5])))

        if dependent == 'freq':
            beta[0] = np.log(X[0, [0]] / X[0, [1]])
        elif dependent == 'sev':
            beta[0] = np.log(np.sum(X[:, [0]]) / np.sum(X[:, [1]]))

        grad = grad_func(beta)
        A = hess_ninv(beta)
        epsilon = 1e-8
        if model != 'NB2':
            lda_step = .1
        else:
            lda_step = 1

        # Estimation algorithm:
        def beta_update(beta, lda_step, A, grad):
            beta_prime = beta + lda_step * A @ grad
            return beta_prime

        start_time = time.perf_counter()
        while True:
            if np.all(np.absolute(grad) < np.ones(np.shape(grad)) * epsilon):
                print('Convergence attained, model ' + model + ', claim type ' + claim_type)
                print('Ellapsed time: ', (time.perf_counter() - start_time)/60)
                break
                    
            beta_prime = beta_update(beta, lda_step, A, grad)
            beta = beta_prime
            grad = grad_func(beta)
            A = hess_ninv(beta)
            sys.stdout.write('Current grad norm: %1.6g \r' % (np.linalg.norm(grad)))
            sys.stdout.flush()

        self.model = model
        self.claim_type = claim_type
        self.beta = beta
        self.LL = LL_func(beta)
        self.var = A
        self.std = np.sqrt(np.diag(A))[:, np.newaxis]
        self.z_stat = beta / self.std

    def save_results(self, keys=None):
        if keys != None:
            for key in keys:
                res_dict[key] = self.key
        else:
            res_dict = {'beta': self.beta, 'LL': self.LL, 'var': self.var, 'std': self.std, 'z_stat': self.z_stat}

        save_results(res_dict, self.model, self.claim_type)


class Testing:
    def __init__(self, model, claim_type):
        self.model = model
        X_dict = file_load(claim_type + '_dictionary.pkl')
        res = grab_results(model, claim_type)
        self.beta = res['beta']
        self.LL = res['LL']
        self.var = res['var']
        self.std = res['std']
        self.z_stat = res['z_stat']
        self.X = X_dict

    def dispersion(self):
        if self.model != 'Poisson':
            print('Method only available for Poisson model')
            return

        aux_phi = 0
        aux_alpha = 0
        aux_dscore1 = 0
        aux_dscore2 = 0
        aux_dscore3 = 0
        n = 0 
        for key in self.X.keys():
            aux_mu = np.exp(np.array([1] + [float(i) for i in list(key)]) @ self.beta)
            mu = self.X[key][:, [2]] * aux_mu
            n += np.shape(self.X[key])[0]
            aux_phi += np.sum((self.X[key][:, [1]] - mu)**2 / mu)
            aux_alpha += np.sum(((self.X[key][:, [1]] - mu)**2 - mu)/ mu**2)
            aux_dscore1 += np.sum((self.X[key][:, [1]] - mu)**2 - mu) / (2 * np.sum(mu**2))**.5
            aux_dscore2 += np.sum((self.X[key][:, [1]] - mu)**2 - self.X[key][:, [1]]) / (2 * np.sum(mu**2))**.5
            aux_dscore3 += np.sum(((self.X[key][:, [1]] - mu)**2 - self.X[key][:, [1]])/ mu)

        df = n - len(self.beta)
        self.phi = aux_phi / df
        self.alpha = aux_alpha / df
        self.dscore1 = aux_dscore1
        self.dscore2 = aux_dscore2
        self.dscore3 = aux_dscore3 / (2*n)**.5



if __name__ == '__main__':
    for model in {'NB2'}:
        for claim_type in ('rcd',):
            print('Estimation: ' + model + ' ' + claim_type)
            x = Estimation(model, claim_type)
            x.save_results()
