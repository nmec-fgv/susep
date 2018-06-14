#########################################################################
## Regression models of counts and claims data for auto insurance cost ##
#########################################################################


import os
import pickle
import shelve
import numpy as np
import sympy
import pdb


# Lower bound parameters for logs and ratios, plus precision parameter:

lb_log = 1e-323
lb_ratio = 1e-308
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

def save_results(res_dict, model, coverage, period, aa):
    db_file = data_dir + model + '_' + coverage + '.db'
    db = shelve.open(db_file)
    db[period+aa] = res_dict
    db.close()
    print('Results for model ' + model + ', coverage ' + coverage + ' period ' + period + aa + ' saved in db file')

    return

def grab_results(model, coverage, period, aa, keys=None):
    db_file = data_dir + model + '_' + coverage + '.db'
    if not os.path.exists(db_file): 
        raise Exception('File ' + dbfile + ' not found')

    db = shelve.open(db_file)
    if keys == None:
        res = db[period+aa]
    else:
        res = {}
        for key in keys:
            res[key] = db[period+aa][key]
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
    '''
    def __init__(self, model, claim_type):
        if claim_type not in {'casco', 'rcd'} or model not in {'Poisson', 'Gamma'}:
            raise Exception('Model or claim_type provided not in permissible set')

        if model in {'Poisson'}:
            dependent = 'freq'
        elif model in {'Gamma'}:
            dependent = 'sev'

        X = file_load(dependent + '_' + claim_type + '_matrix.pkl')

        if model == 'Poisson':#### remove later
            X = np.delete(X, -4, 1) 
        elif model == 'Gamma':#### remove later
            X = np.delete(X, np.s_[-4:], 1)
#            X = np.delete(X, np.s_[10000:], axis=0)
            X = np.insert(X, 2, 1, axis=1)

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

        # Initial guesses and stoping parameter:
        beta = np.zeros(np.shape(X)[1] - 2)[:, np.newaxis]
        if dependent == 'freq':
            beta[0] = np.log(X[0, [0]] / X[0, [1]])
        elif dependent == 'sev':
            beta[0] = np.log(np.sum(X[:, [0]]) / np.sum(X[:, [1]]))

        grad = grad_func(beta)
        LL = LL_func(beta)
        A = hess_ninv(beta)
        epsilon = 1e-8

        # Estimation algorithm:
        def beta_update(beta, lda_step, A, grad):
            beta_prime = beta + lda_step * A @ grad
            return beta_prime

        while True:
            if np.all(np.absolute(grad) < np.ones(np.shape(grad)) * epsilon):
                    print('convergence successful')
                    break
                    
            lda_step = .1
            beta_prime = beta_update(beta, lda_step, A, grad)
            LL_prime = LL_func(beta_prime)
            beta = beta_prime
            grad = grad_func(beta)
            A = hess_ninv(beta)
            LL = LL_prime
            print('Current grad eval: ', grad)

        self.beta = beta
        self.LL = LL
        self.var = A


if __name__ == '__main__':
    fcasco = Estimation('Gamma', 'rcd')
