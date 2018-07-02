#########################################################################
## Regression models of counts and claims data for auto insurance cost ##
#########################################################################

import os
import sys
import pickle
import shelve
import numpy as np
import scipy.special as ss
import pdb
import time

# Lower bound and precision parameter:

lb_alpha = 1e-77
prec_param = 1e-8

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

def save_results_db(res, prefix, model, claim_type):
    db_file = data_dir + prefix + '_results_' + claim_type + '.db'
    db = shelve.open(db_file)
    db[model] = res
    db.close()
    print(prefix + ' results from ' + model + ' ' + claim_type + ', saved in db file')
    return

def save_results_pkl(res, prefix, model, claim_type):
    try:
        os.remove(data_dir + prefix + '_results_' + claim_type + '.pkl')
    except OSError:
        pass

    with open(data_dir + prefix + '_results_' + claim_type + '.pkl') as filename:
        pickle.dump(res, filename)

    print(prefix + ' results from ' + model + ' ' + claim_type + ', saved in pkl file')

def grab_results_db(prefix, model, claim_type, keys=None):
    db_file = data_dir + prefix + '_results_' + claim_type + '.db'
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
            def grad_func(beta):
                '''
                Gradient of Log-likelihood for Poisson regression model
                [y_i - exposure_i * exp(x_i'beta)] * x_i
                '''
            
                aux_vec = X[:, [0]] - X[:, [1]] * np.exp(X[:, 2:] @ beta)
                res = (aux_vec.T @ X[:, 2:]).T
                return res
    
            def hess_ninv(beta):
                '''
                Inverse of negative Hessian of Poisson loglikelihood
                inv mu_i * x_i * x_i'
                '''
    
                res = np.linalg.inv(X[:, 2:].T @ (X[:, [1]] * np.exp(X[:, 2:] @ beta) * X[:, 2:]))
                return res

        elif model == 'NB2':
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
                inv (mu_i/(1+alpha*mu_i)] * x_i * x_i'
                inv (1/alpha^4)*[ln(1+alpha*mu_i)-sum_{j=0}^{y_i-1}1/(j+alpha^-1)]^2+mu_i/(alpha**2*(1+alpha*mu_i)
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
            def grad_func(beta):
                '''
                Gradient of Log-likelihood for Gamma regression model
                [(sum c_ik/exp(x_i'beta)-n_i] * x_i
                n_i*[-digamma(nu)+ln(nu)+1-x_i'beta+ln(c_bar)]-sum c_ik*exp(-x_i'beta)
                '''
            
                aux_beta = X[:, [0]] * np.exp(-1 * X[:, 2:] @ beta[:-1]) - X[:, [1]] 
                aux_beta = (aux_beta.T @ X[:, 2:]).T
                aux_nu = np.sum(X[:, [1]] * (-ss.digamma(beta[-1]) + np.log(beta[-1]) + 1 - (X[:, 2:] @ beta[:-1]) + np.log(X[:, [0]] / X[:, [1]])) - X[:, [0]] * np.exp(-1 * X[:, 2:] @ beta[:-1]))
                res = np.vstack((aux_beta, aux_nu))
                return res
    
            def hess_ninv(beta):
                '''
                Inverse of negative Hessian of Gamma loglikelihood
                inv [sum c_ik/exp(x_i'beta] * x_i * x_i'
                inv n_i*[polygamma(1,nu)-nu^(-1)]
                '''
    
                aux_beta = np.linalg.inv(X[:, 2:].T @ (X[:, [0]] * np.exp(-1 * X[:, 2:] @ beta[:-1]) * X[:, 2:]))
                aux_nu = (np.sum(X[:, [1]] * (ss.polygamma(1, beta[-1]) - beta[-1]**(-1))))**(-1)
                res = np.hstack((aux_beta, np.zeros(np.shape(aux_beta)[0])[:, np.newaxis]))
                res = np.vstack((res, np.concatenate((np.zeros(np.shape(res)[1]-1), [aux_nu]))))
                return res

        elif model == 'InvGaussian':
            def grad_func(beta):
                '''
                Gradient of Log-likelihood for InvGaussian regression model
                [sum c_ik*exp(-2*x_i'beta) - n_i*exp(-x_i'beta)] * x_i
                -n_i/2sigma^2 + 1/sigma^4*[sum c_ik*exp(-2x_i'beta)/2 - n_i*exp(-x_i'beta) + 1/2*sum c_ik]
                '''
            
                aux_beta = X[:, [0]] * np.exp(-2 * X[:, 2:] @ beta[:-1]) - X[:, [1]] * np.exp(-1 * X[:, 2:] @ beta[:-1])
                aux_beta = (aux_beta.T @ X[:, 2:]).T
                aux_sigma2 = np.sum(-X[:, [1]]/(2 * beta[-1]) + beta[-1]**(-2) * (.5 * X[:, [0]] * np.exp(-2 * X[:, 2:] @ beta[:-1]) - X[:, [1]] * np.exp(-1 * X[:, 2:] @ beta[:-1]) + (2 * X[:, [0]])**(-1)))
                res = np.vstack((aux_beta, aux_sigma2))
                return res
    
            def hess_ninv(beta):
                '''
                Inverse of negative Hessian of InvGaussian loglikelihood
                [2 * sum c_ik * exp(-2*x_i'beta) - n_i * exp(-x_i'beta)] * x_i * x_i'
                -n_i/2sigma^4 + 2/sigma^6*[sum c_ik*exp(-2x_i'beta)/2 - n_i*exp(-x_i'beta) + 1/2*sum c_ik]
                '''
    
                aux_beta = np.linalg.inv(X[:, 2:].T @ ((2 * X[:, [0]] * np.exp(-2 * X[:, 2:] @ beta[:-1]) - X[:, [1]] * np.exp(-1 * X[:, 2:] @ beta[:-1])) * X[:, 2:]))
                aux_sigma2 = (np.sum(-X[:, [1]]/(2 * beta[-1]**2) + (2/beta[-1]**(3)) * (.5 * X[:, [0]] * np.exp(-2 * X[:, 2:] @ beta[:-1]) - X[:, [1]] * np.exp(-1 * X[:, 2:] @ beta[:-1]) + (2 * X[:, [0]])**(-1))))**(-1)
                res = np.hstack((aux_beta, np.zeros(np.shape(aux_beta)[0])[:, np.newaxis]))
                res = np.vstack((res, np.concatenate((np.zeros(np.shape(res)[1]-1), [aux_sigma2]))))
                return res

        # Initial guesses and stoping parameter:
        beta = np.zeros(np.shape(X)[1] - 2)[:, np.newaxis]
        if model == 'NB2':
            beta = np.vstack((beta, np.array([.5])))
        elif model == 'Gamma':
            beta = np.vstack((beta, np.array([2])))
        elif model == 'InvGaussian':
            beta = np.vstack((beta, np.array([2])))

        if dependent == 'freq':
            beta[0] = np.log(X[0, [0]] / X[0, [1]])
        elif dependent == 'sev':
            beta[0] = np.log(np.sum(X[:, [0]]) / np.sum(X[:, [1]]))

        grad = grad_func(beta)
        A = hess_ninv(beta)
        epsilon = prec_param
        if model not in {'NB2'}:
            lda_step = .1
        else:
            lda_step = 1

        # Newton-Raphson algorithm:
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
        self.var = A
        self.std = np.sqrt(np.diag(A))[:, np.newaxis]
        self.z_stat = beta / self.std

    def save_estimation_results(self, keys=None):
        if keys != None:
            for key in keys:
                res_dict[key] = self.key
        else:
            res_dict = {'beta': self.beta, 'var': self.var, 'std': self.std, 'z_stat': self.z_stat}

        prefix = 'overall'
        save_results_db(res_dict, prefix, self.model, self.claim_type)


class Stdout:
    '''
    Provides standard output for GLMs.

    Three presistent files w/ following statistics:
    individual_results_xxx.pkl: deviances and chis
    cell_results_xxx.db: exposure, y_bar, mu_hat, D^L
    overall_results_xxx.db: LL, D, Chi^2
    '''

    def __init__(self, model, claim_type):
        self.model = model
        self.claim_type = claim_type
        X_dict = file_load(claim_type + '_dictionary.pkl')
        prefix = 'overall'
        keys = ('beta',)
        res = grab_results_db(prefix, model, claim_type, keys)
        beta = res['beta']
        cell_res = np.empty([len(X_dict), 3])
        if model == 'Poisson':
            def LL_func(X, std_mu, extra_param):
                '''
                Log-likelihood: sum_i -exposure_i * exp(x_i'beta) + y_i * ln(exposure_i * exp(x_i'beta)) - ln y_i!
                '''

                res = np.sum(- X[:, [2]] * std_mu + X[:, [1]] * np.log(X[:, [2]] * std_mu) - np.log(ss.factorial(X[:, [1]])))
                return res

        elif model == 'NB2':
            def LL_func(X, std_mu, extra_param):
                '''
                Log-likelihood: sum_i ln(gamma(y_i + alpha^(-1))/gamma(alpha^(-1))) -ln y_i! -(y_i + alpha^(-1))*ln(alpha^(-1) + exposure_i*exp(x_i'beta)) + alpha^(-1)*ln(alpha^(-1)) + y_i*ln(exposure_i*exp(x_i'beta)) 
                '''
            
                inv_alpha = extra_param**(-1)
                res = np.sum(np.log(ss.gamma(X[:, [1]] + inv_alpha) / ss.gamma(inv_alpha)) - np.log(ss.factorial(X[:, [1]])) - (X[:, [1]] + inv_alpha) * np.log(inv_alpha + X[:, [2]] * std_mu) + inv_alpha * np.log(inv_alpha) + X[:, [1]] * np.log(X[:, [2]] * std_mu))
                return res

        elif model == 'Gamma':
            def LL_func(beta):
                '''
                Log-likelihood for Gamma regression model, nu excluded, constant term excluded
                -n_i * x_i'beta -sum c_ik * exp(-x_i'beta) 
                '''
            
                res = np.sum(- X[:, [1]] * X[:, 2:] @ beta - X[:, [0]] * np.exp(X[:, 2:] @ beta))
                return res

        elif model == 'InvGaussian':
            def LL_func(beta):
                '''
                Log-likelihood for InvGaussian regression model, sigma excluded, constant term excluded
                -(sum c_ik/2) * exp(-2*x_i'beta) + n_i * exp(-x_i'beta) 
                '''
            
                res = np.sum(- (X[:, [0]] / 2) * np.exp(-2 * X[:, 2:] @ beta) + X[:, [1]] * np.exp(-1 * X[:, 2:] @ beta))
                return res
            
        LL_sum = 0
        for i, key in enumerate(X_dict.keys()):
            X = X_dict[key]
            # Except for Poisson, other models have two parameters in beta vector:
            if model == 'Poisson':
                std_mu = np.exp(np.array([1] + [float(j) for j in list(key)]) @ beta)
                extra_param = None
            else:
                std_mu = np.exp(np.array([1] + [float(j) for j in list(key)]) @ beta[:-1])
                extra_param = beta[-1]

            LL_sum += LL_func(X, std_mu, extra_param)
            cell_res[i, 0] = np.sum(X[:, [2]])
            cell_res[i, 1] = np.average(X[:, [1]])
            cell_res[i, 2] = std_mu

        self.LL = LL_sum


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
    for model in ('Gamma',):
        for claim_type in ('rcd', 'casco'):
            print('Estimation: ' + model + ' ' + claim_type)
            x = Estimation(model, claim_type)
            x.save_estimation_results()
#            x = Stdout(model, claim_type)
