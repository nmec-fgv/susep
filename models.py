#########################################################################
## Regression models of counts and claims data for auto insurance cost ##
#########################################################################

import os
import sys
import pickle
import shelve
import numpy as np
import scipy.special as sp
import scipy.stats as st
import pdb
import time

# Lower bound and precision parameter:

lb_log = 1e-323
lb_ratio = 1e-308
lb_alpha = 1e-77
lb_sigma2 = 1e-102
prec_param = 1e-8

# Data directory:

data_dir = 'persistent/'

# Auxiliary functions:

def file_load(filename):
    try:
        os.path.exists(data_dir + filename)
        with open(data_dir + filename, 'rb') as cfile:
            x = pickle.load(cfile)
    except:
        print('File ' + filename + ' not found')

    return x

def save_results_db(res_dict, prefix, model, claim_type):
    db_file = data_dir + prefix + '_results_' + claim_type + '.db'
    with shelve.open(db_file, writeback=True) as db:
        if model in db.keys():
            for key in res_dict.keys():
                db[model][key] = res_dict[key]
        else:
            db[model] = res_dict

    print(prefix + ' results from ' + model + ' ' + claim_type + ', saved in db file')
    return

def save_results_pkl(res, prefix, model, claim_type):
    try:
        os.remove(data_dir + prefix + '_results_' + model + '_' + claim_type + '.pkl')
    except OSError:
        pass

    with open(data_dir + prefix + '_results_' + model + '_' + claim_type + '.pkl', 'wb') as filename:
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
    model - specification of density of random component

    Methods:
    --------
    save_results
    '''
    def __init__(self, model, claim_type):
        if claim_type not in {'casco', 'rcd'} or model not in {'Poisson', 'NB2', 'Logit', 'Probit', 'C-loglog', 'Gamma', 'InvGaussian'}:
            raise Exception('Model or claim_type provided not in permissible set')

        if model in {'Poisson', 'NB2', 'Logit', 'Probit', 'C-loglog'}:
            dependent = 'freq'
        elif model in {'Gamma', 'InvGaussian'}:
            dependent = 'sev'

        X = file_load(dependent + '_' + claim_type + '_matrix.pkl')
        if model == 'Poisson':
            def grad_func(X, beta):
                '''
                Gradient of Log-likelihood for Poisson regression model
                [y_i - exposure_i * exp(x_i'beta)] * x_i
                '''
            
                aux_vec = X[:, [0]] - X[:, [1]] * np.exp(X[:, 2:] @ beta)
                res = (aux_vec.T @ X[:, 2:]).T
                return res
    
            def hess_ninv(X, beta):
                '''
                Inverse of negative Hessian of Poisson loglikelihood
                inv mu_i * x_i * x_i'
                '''
    
                res = np.linalg.inv(X[:, 2:].T @ (X[:, [1]] * np.exp(X[:, 2:] @ beta) * X[:, 2:]))
                return res

        elif model == 'NB2':
            def grad_func(X, beta):
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
    
            def hess_ninv(X, beta):
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

        if model == 'Logit':
            def grad_func(X, beta):
                '''
                Gradient of Log-likelihood for Logit regression model
                [y_i - m_i * exp(x_i'beta)/(1+exp(x_i'beta)] * x_i
                '''
            
                aux_vec = X[:, [0]] - X[:, [1]] * np.exp(X[:, 2:] @ beta) / (1 + np.exp(X[:, 2:] @ beta))
                res = (aux_vec.T @ X[:, 2:]).T
                return res
    
            def hess_ninv(X, beta):
                '''
                Inverse of negative expected Hessian of Logit loglikelihood
                inv m_i * exp(x_i'beta)/(1+exp(x_i'beta)**2 * x_i * x_i'
                '''
    
                res = np.linalg.inv(X[:, 2:].T @ (X[:, [1]] * np.exp(X[:, 2:] @ beta) / (1 + np.exp(X[:, 2:] @ beta))**2 * X[:, 2:]))
                return res

        if model == 'Probit':
            def grad_func(X, beta):
                '''
                Gradient of Log-likelihood for Probit regression model
                [((y_i - m_i * Phi(x_i'beta))/(Phi(x_i'beta)(1-Phi(x_i'beta)) * phi(x_i'beta)] * x_i
                '''
            
                Phi = st.norm.cdf(X[:, 2:] @ beta)
                phi = st.norm.pdf(X[:, 2:] @ beta)
                aux_vec = ((X[:, [0]] - X[:, [1]] * Phi) / (Phi * (1 - Phi))) * phi
                res = (aux_vec.T @ X[:, 2:]).T
                return res
    
            def hess_ninv(X, beta):
                '''
                Inverse of negative expected Hessian of Probit loglikelihood
                inv m_i * phi(x_i'beta)**2 / (Phi(x_i'beta)(1-Phi(x_i'beta)) * x_i * x_i'
                '''
    
                Phi = st.norm.cdf(X[:, 2:] @ beta)
                phi = st.norm.pdf(X[:, 2:] @ beta)
                aux_vec = (X[:, [1]] * phi**2) / (Phi * (1 - Phi))
                res = np.linalg.inv(X[:, 2:].T @ (aux_vec * X[:, 2:]))
                return res

        if model == 'C-loglog':
            def grad_func(X, beta):
                '''
                Gradient of Log-likelihood for Complementary log-log regression model
                {[y_i * (1 - exp(-exp(x_i'beta)))^(-1) - m_i] * exp(x_i'beta)} * x_i
                '''
            
                aux_vec = X[:, [0]] - X[:, [1]] * np.exp(X[:, 2:] @ beta)
                res = (aux_vec.T @ X[:, 2:]).T
                return res
    
            def hess_ninv(X, beta):
                '''
                Inverse of negative Hessian of Poisson loglikelihood
                inv {m_i * exp(x_i'beta)^2*exp(-exp(x_i'beta))*[1-exp(-exp(x_i'beta))]^(-1)} * x_i * x_i'
                '''
    
                aux_vec = X[:, [1]] * np.exp(X[:, 2:] @ beta)**2 * np.exp(-np.exp(X[:, 2:] @ beta)) * (1 - np.exp(-np.exp(X[:, 2:] @ beta)))**(-1)
                res = np.linalg.inv(X[:, 2:].T @ (aux_vec * X[:, 2:]))
                return res

        elif model == 'Gamma':
            def grad_func(X, beta):
                '''
                Gradient of Log-likelihood for Gamma regression model
                [(y_i/exp(x_i'beta)-1] * x_i
                -y*exp(-x_i'beta)-x_i'beta+ln(y)+ln(nu)+1-digamma(nu)
                '''
            
                beta[-1] = np.maximum(beta[-1], lb_log)
                aux_beta = X[:, [0]] * np.exp(-1 * X[:, 1:] @ beta[:-1]) - 1 
                aux_beta = (aux_beta.T @ X[:, 1:]).T
                aux_nu = np.sum(- X[:, [0]] * np.exp(-1 * X[:, 1:] @ beta[:-1]) - (X[:, 1:] @ beta[:-1]) + np.log(X[:, [0]]) + np.log(beta[-1]) + 1 - sp.digamma(beta[-1]))
                res = np.vstack((aux_beta, aux_nu))
                return res
    
            def hess_ninv(X, beta):
                '''
                Inverse of negative Hessian of Gamma loglikelihood
                inv [y_i/exp(x_i'beta] * x_i * x_i'
                inv [polygamma(1,nu)-nu^(-1)]
                '''
    
                beta[-1] = np.maximum(beta[-1], lb_ratio)
                aux_beta = np.linalg.inv(X[:, 1:].T @ (X[:, [0]] * np.exp(-1 * X[:, 1:] @ beta[:-1]) * X[:, 1:]))
                aux_nu = (np.sum(np.ones(len(X))[:, np.newaxis] * sp.polygamma(1, beta[-1]) - beta[-1]**(-1)))**(-1)
                res = np.hstack((aux_beta, np.zeros(np.shape(aux_beta)[0])[:, np.newaxis]))
                res = np.vstack((res, np.concatenate((np.zeros(np.shape(res)[1]-1), [aux_nu]))))
                return res

        elif model == 'InvGaussian':
            def grad_func(X, beta):
                '''
                Gradient of Log-likelihood for InvGaussian regression model
                [y_i*exp(-2*x_i'beta) - exp(-x_i'beta)] * x_i
                -1/2*sigma^2 + 1/sigma^4*(y_i*exp(-2x_i'beta)/2 - exp(-x_i'beta) + 1/2*y_i)
                '''
            
                beta[-1] = np.maximum(beta[-1], lb_sigma2)
                aux_beta = X[:, [0]] * np.exp(-2 * X[:, 1:] @ beta[:-1]) - np.exp(-1 * X[:, 1:] @ beta[:-1])
                aux_beta = (aux_beta.T @ X[:, 1:]).T
                aux_sigma2 = np.sum(-(2 * beta[-1])**(-1) + beta[-1]**(-2) * (.5 * X[:, [0]] * np.exp(-2 * X[:, 1:] @ beta[:-1]) - np.exp(-1 * X[:, 1:] @ beta[:-1]) + (2 * X[:, [0]])**(-1)))
                res = np.vstack((aux_beta, aux_sigma2))
                return res
    
            def hess_ninv(X, beta):
                '''
                Inverse of negative Hessian of InvGaussian loglikelihood
                [2 * y_i * exp(-2*x_i'beta) - exp(-x_i'beta)] * x_i * x_i'
                -1/2*sigma^4 + 2/sigma^6*(y_i*exp(-2x_i'beta)/2 - exp(-x_i'beta) + 1/2*y_i)
                '''
    
                beta[-1] = np.maximum(beta[-1], lb_sigma2)
                aux_beta = np.linalg.inv(X[:, 1:].T @ ((2 * X[:, [0]] * np.exp(-2 * X[:, 1:] @ beta[:-1]) - np.exp(-1 * X[:, 1:] @ beta[:-1])) * X[:, 1:]))
                aux_sigma2 = (np.sum(-(2 * beta[-1]**2)**(-1) + (2/beta[-1]**(3)) * (.5 * X[:, [0]] * np.exp(-2 * X[:, 1:] @ beta[:-1]) - np.exp(-1 * X[:, 1:] @ beta[:-1]) + (2 * X[:, [0]])**(-1))))**(-1)
                res = np.hstack((aux_beta, np.zeros(np.shape(aux_beta)[0])[:, np.newaxis]))
                res = np.vstack((res, np.concatenate((np.zeros(np.shape(res)[1]-1), [aux_sigma2]))))
                return res

        # Initial guesses and stoping parameter:
        if dependent == 'freq':
            beta = np.zeros(np.shape(X)[1] - 2)[:, np.newaxis]
        elif dependent == 'sev':
            beta = np.zeros(np.shape(X)[1] - 1)[:, np.newaxis]

        if model in {'NB2', 'Gamma', 'InvGaussian'}:
            beta = np.vstack((beta, np.array([.5])))

        if dependent == 'freq':
            if model in {'Poisson', 'NB2'}:
                beta[0] = np.log(X[0, [0]] / X[0, [1]])
            elif model in {'Logit', 'Probit', 'C-loglog'}:
                beta[0] = .2
        elif dependent == 'sev':
            beta[0] = 1

        grad = grad_func(X, beta)
        A = hess_ninv(X, beta)
        epsilon = prec_param
        lda_step = 1

        # Newton-Raphson algorithm:
        def beta_update(beta, lda_step, A, grad):
            beta_prime = beta + lda_step * A @ grad
            return beta_prime

        start_time = time.perf_counter()
        print('Estimation: ' + model + ' ' + claim_type)
        while True:
            if np.all(np.absolute(grad) < np.ones(np.shape(grad)) * epsilon):
                print('Convergence attained, model ' + model + ', claim type ' + claim_type)
                print('Ellapsed time: ', (time.perf_counter() - start_time)/60)
                break
                    
            beta_prime = beta_update(beta, lda_step, A, grad)
            beta = beta_prime
            grad = grad_func(X, beta)
            A = hess_ninv(X, beta)
            sys.stdout.write('Current grad norm: %1.6g \r' % (np.linalg.norm(grad)))
            sys.stdout.flush()

        self.model = model
        self.claim_type = claim_type
        self.beta = beta
        self.var = A
        self.std = np.sqrt(np.diag(A))[:, np.newaxis]
        self.z_stat = beta / self.std
        if dependent == 'freq':
            self.y_bar = np.sum(X[:, [0]]) / np.sum(X[:, [1]])
        elif dependent == 'sev':
            self.y_bar = np.average(X[:, [0]])

    def save_estimation_results(self, keys=None):
        if keys != None:
            for key in keys:
                res_dict[key] = self.key
        else:
            res_dict = {'beta': self.beta, 'var': self.var, 'std': self.std, 'z_stat': self.z_stat, 'y_bar': self.y_bar}

        prefix = 'overall'
        save_results_db(res_dict, prefix, self.model, self.claim_type)


class Stdout:
    '''
    Provides standard output for GLMs.
    Reads freq/sev_<claim_type>_dictionary.pkl, where keys index vector of dummies and:
    Frequency:
    column 0 = claims count, y_i
    column 1 = exposure_i
    Severity:
    column 0 = claim cost, y_i

    Three persistent files w/ following statistics:
    individual_results_xxx.pkl: deviances and chis
    cell_results_xxx.db: # obs, y_bar, mu_hat, D^L, exposure (freq only)
    overall_results_xxx.db: LL, D, Pearson, Chi^2
    
    Standard outpu is different for models w/ Binomial distribution:
    counts and exposure are aggregated within cells before computation of likelihood and residuals
    '''

    def __init__(self, model, claim_type):
        self.model = model
        self.claim_type = claim_type
        if model in {'Poisson', 'NB2', 'Logit', 'Probit', 'C-loglog'}:
            model_type = 'freq'
        elif model in {'Gamma', 'InvGaussian'}:
            model_type = 'sev'

        self.model_type = model_type
        X_dict = file_load(model_type + '_' + claim_type + '_dict.pkl')
        res = grab_results_db('overall', model, claim_type)
        self.beta = res['beta']
        self.var = res['var']
        self.std = res['std']
        self.z_stat = res['z_stat']
        self.y_bar = res['y_bar']
        ind_res = {}
        if model_type == 'freq':
            cell_res = np.empty([len(X_dict), 6])
        elif model_type == 'sev':
            cell_res = np.empty([len(X_dict), 5])

        if model == 'Poisson':
            def LL_func(X, mu, extra_param):
                '''
                Log-likelihood: sum_i -exposure_i * exp(x_i'beta) + y_i * ln(exposure_i * exp(x_i'beta)) - ln y_i!
                '''

                res = np.sum(- X[:, [1]] * mu + X[:, [0]] * np.log(X[:, [1]] * mu) - np.log(sp.factorial(X[:, [0]])))
                return res

            def deviance(X, mu, extra_param, y_bar):
                '''
                deviance^2 = y_i * ln(y_i/mu_i) - (y_i - mu_i)
                '''

                aux_dev = np.zeros(np.shape(X[:, [0]]))
                aux_dev_y_bar = np.zeros(np.shape(X[:, [0]]))
                index = np.where(X[:, [0]] > 0)[0]
                aux_dev[index] = X[:, [0]][index] * np.log(X[:, [0]][index] / (X[:, [1]][index] * mu))
                aux_dev_y_bar[index] = X[:, [0]][index] * np.log(X[:, [0]][index] / (X[:, [1]][index] * y_bar))
                aux2_dev = X[:, [0]] - X[:, [1]] * mu
                aux2_dev_y_bar = X[:, [0]] - X[:, [1]] * y_bar
                dev_local = 2 * np.sum(aux_dev - aux2_dev)
                dev_y_bar_local = 2 * np.sum(aux_dev_y_bar - aux2_dev_y_bar)
                index2 = np.where(X[:, [0]] - X[:, [1]] * mu < 0)[0]
                dev_is = (2*(aux_dev - aux2_dev))**.5
                dev_is[index2] = -1 * dev_is[index2]
                return (dev_is, dev_local, dev_y_bar_local)

            def Pearson(X, mu, extra_param):
                '''(y_i-mu_i) / mu_i^.5'''

                Pearson_is = (X[:, [0]] - X[:, [1]] * mu) / (X[:, [1]] * mu)**.5
                Pearson_local = np.sum(Pearson_is**2)
                return (Pearson_is, Pearson_local)

        elif model == 'NB2':
            def LL_func(X, mu, extra_param):
                '''
                Log-likelihood: sum_i ln(gamma(y_i + alpha^(-1))/gamma(alpha^(-1))) -ln y_i! -(y_i + alpha^(-1))*ln(alpha^(-1) + exposure_i*exp(x_i'beta)) + alpha^(-1)*ln(alpha^(-1)) + y_i*ln(exposure_i*exp(x_i'beta)) 
                '''
            
                inv_alpha = extra_param**(-1)
                res = np.sum(np.log(sp.gamma(X[:, [0]] + inv_alpha) / sp.gamma(inv_alpha)) - np.log(sp.factorial(X[:, [0]])) - (X[:, [0]] + inv_alpha) * np.log(inv_alpha + X[:, [1]] * mu) + inv_alpha * np.log(inv_alpha) + X[:, [0]] * np.log(X[:, [1]] * mu))
                return res

            def deviance(X, mu, extra_param, y_bar):
                '''
                deviance^2 = y_i * ln(y_i/mu_i) - (y_i + alpha^(-1))*ln((y_i + alpha^(-1))/(mu_i + alpha^(-1)))
                '''

                inv_alpha = extra_param**(-1)
                aux_dev = np.zeros(np.shape(X[:, [0]]))
                aux_dev_y_bar = np.zeros(np.shape(X[:, [0]]))
                index = np.where(X[:, [0]] > 0)[0]
                aux_dev[index] = X[:, [0]][index] * np.log(X[:, [0]][index] / (X[:, [1]][index] * mu))
                aux_dev_y_bar[index] = X[:, [0]][index] * np.log(X[:, [0]][index] / (X[:, [1]][index] * y_bar))
                aux2_dev = (X[:, [0]] + inv_alpha) * np.log((X[:, [0]] + inv_alpha) / (X[:, [1]] * mu + inv_alpha))
                aux2_dev_y_bar = (X[:, [0]] + inv_alpha) * np.log((X[:, [0]] + inv_alpha) / (X[:, [1]] * y_bar + inv_alpha))
                dev_local = 2 * np.sum(aux_dev - aux2_dev)
                dev_y_bar_local = 2 * np.sum(aux_dev_y_bar - aux2_dev_y_bar)
                index2 = np.where(X[:, [0]] - X[:, [1]] * mu < 0)[0]
                dev_is = (2*(aux_dev - aux2_dev))**.5
                dev_is[index2] = -1 * dev_is[index2]
                return (dev_is, dev_local, dev_y_bar_local)

            def Pearson(X, mu, extra_param):
                '''(y_i-mu_i) / (mu_i+alpha*mu_i^2)^.5'''

                Pearson_is = (X[:, [0]] - X[:, [1]] * mu) / (X[:, [1]] * mu + extra_param * (X[:, [1]] * mu)**2)**.5
                Pearson_local = np.sum(Pearson_is**2)
                return (Pearson_is, Pearson_local)

        elif model in {'Logit', 'Probit', 'C-loglog'}:
            def LL_func(X, mu, extra_param):
                '''
                Log-likelihood: sum_i y_i * ln(mui_i/(1-mu_i)) + m_i * ln(1-mu_i)
                '''

                y = np.sum(X[:, [0]])
                m = np.sum(X[:, [1]])
                res = np.sum(y * np.log(mu / (1 - mu)) + m * np.log(1 - mu))
                return res

            def deviance(X, mu, extra_param, y_bar):
                '''
                deviance^2 = y_i * ln(y_i/mu_i) + (m_i - y_i) * ln((m_i - y_i)/(m_i-mu_i))
                '''

                y = np.sum(X[:, [0]])
                m = np.sum(X[:, [1]])
                if y > 0:
                    dev_local = 2 * (y * np.log(y / mu) + (m - y) * np.log((m - y) / (m - mu)))
                    dev_y_bar_local = 2 * (y * np.log(y / y_bar) + (m - y) * np.log((m - y) / (m - y_bar)))

                else:
                    dev_local = 2 * (m * np.log(m / (m - mu)))
                    dev_y_bar_local = 0

                if y >= mu:
                    dev_is = dev_local**.5
                else:
                    dev_is = - dev_local**.5

                return (dev_is, dev_local, dev_y_bar_local)

            def Pearson(X, mu, extra_param):
                '''(y_i/m_i - mu_i)^2 / mu_i*(1-mu_i)'''

                y = np.sum(X[:, [0]])
                m = np.sum(X[:, [1]])
                Pearson_is = (y / m - mu) / (mu * (1 - mu))**.5
                Pearson_local = Pearson_is**2
                return (Pearson_is, Pearson_local)

        elif model == 'Gamma':
            def LL_func(X, mu, extra_param):
                '''
                Log-likelihood: sum_i -nu*(y_i/mu_i)-nu*ln(mu_i)+nu*ln(y_i)+nu*ln(nu)-ln(y_i)-ln(gamma(nu))
                '''
            
                nu = extra_param
                res = np.sum(- nu * (X[:, [0]] / mu) - nu * np.log(mu) + nu * np.log(X[:, [0]]) + nu * np.log(nu) - np.log(X[:, [0]]) - np.log(sp.gamma(nu)))
                return res

            def deviance(X, mu, extra_param, y_bar):
                '''
                deviance^2 = -ln(y_i/mu_i) + (y_i - mu_i) / mu_i
                '''

                aux_dev = - np.log(X / mu) + (X - mu) / mu
                aux_dev_y_bar = - np.log(X / y_bar) + (X - y_bar) / y_bar
                dev_local = 2 * np.sum(aux_dev)
                dev_y_bar_local = 2 * np.sum(aux_dev_y_bar)
                index = np.where(X - mu < 0)[0]
                dev_is = (2*aux_dev)**.5
                dev_is[index] = -1 * dev_is[index]
                return (dev_is, dev_local, dev_y_bar_local)

            def Pearson(X, mu, extra_param):
                '''(y_i-mu_i) / (mu_i^2)^.5'''

                Pearson_is = (X - mu) / mu
                Pearson_local = np.sum(Pearson_is**2)
                return (Pearson_is, Pearson_local)

        elif model == 'InvGaussian':
            def LL_func(X, mu, extra_param):
                '''
                Log-likelihood: sum_i -.5*ln(2*pi*sigma2)-.5*ln(y_i^3)-y_i/2*sigma2*mu^2+1/sigma2*mu_i-(1/2*sigma2*y_i
                '''
            
                sigma2 = extra_param
                res = np.sum(- .5 * np.log(2 * np.pi * sigma2) - .5 * np.log(X**3) - X / (2 * sigma2 * mu**2) + (sigma2 * mu)**(-1) - (2 * sigma2 * X)**(-1))
                return res
            
            def deviance(X, mu, extra_param, y_bar):
                '''
                deviance^2 = (y_i-mu_i)^2/(mu_i^2*y_i)
                '''

                aux_dev = (X - mu)**2 / (mu**2*X)
                aux_dev_y_bar = (X - y_bar)**2 / (y_bar**2*X)
                dev_local = 2 * np.sum(aux_dev)
                dev_y_bar_local = 2 * np.sum(aux_dev_y_bar)
                index = np.where(X - mu < 0)[0]
                dev_is = (2*aux_dev)**.5
                dev_is[index] = -1 * dev_is[index]
                return (dev_is, dev_local, dev_y_bar_local)

            def Pearson(X, mu, extra_param):
                '''(y_i-mu_i) / (mu_i^3)^.5'''

                Pearson_is = (X - mu) / mu**1.5
                Pearson_local = np.sum(Pearson_is**2)
                return (Pearson_is, Pearson_local)

        LL_sum = 0
        dev_stat_sum = 0
        dev_y_bar_stat_sum = 0
        Pearson_stat_sum = 0
        for i, key in enumerate(X_dict.keys()):
            X = X_dict[key]
            if model == 'Poisson':
                mu = np.exp(np.array([1] + [float(j) for j in list(key)]) @ self.beta)
                extra_param = None
            elif model == 'Logit':
                aux_mu = np.exp(np.array([1] + [float(j) for j in list(key)]) @ self.beta)
                mu = aux_mu / (1 + aux_mu)
                extra_param = None
            elif model == 'Probit':
                aux_mu = np.array([1] + [float(j) for j in list(key)]) @ self.beta
                mu = st.norm.cdf(aux_mu)
                extra_param = None
            elif model == 'C-loglog':
                mu = 1 - np.exp(- np.exp(np.array([1] + [float(j) for j in list(key)]) @ self.beta))
                extra_param = None
            else:
                mu = np.exp(np.array([1] + [float(j) for j in list(key)]) @ self.beta[:-1])
                extra_param = self.beta[-1]

            if np.shape(X)[0] > 0:
                LL_sum += LL_func(X, mu, extra_param)
                (dev_is, dev_local, dev_y_bar_local) = deviance(X, mu, extra_param, self.y_bar)
                (Pearson_is, Pearson_local) = Pearson(X, mu, extra_param)
                ind_res[key] = np.hstack((dev_is, Pearson_is))
                if model_type == 'freq':
                    cell_res[i, 1] = np.average(X[:, [0]] * X[:, [1]])
                elif model_type == 'sev':
                    cell_res[i, 1] = np.average(X[:, [0]])
            else:
                ind_res[key] = np.array([])
                cell_res[i, 1] = 0

            cell_res[i, 0] = len(X)
            cell_res[i, 2] = mu
            cell_res[i, 3] = dev_local
            cell_res[i, 4] = Pearson_local
            if model_type == 'freq':
                cell_res[i, 5] = np.sum(X[:, [1]])

            dev_stat_sum += dev_local
            dev_y_bar_stat_sum += dev_y_bar_local
            Pearson_stat_sum += Pearson_local

        self.ind_res = ind_res
        self.cell_res = cell_res
        self.p_value = st.norm.cdf(abs(self.z_stat))
        self.n = np.sum(cell_res[:, [0]])
        self.k = len(self.beta)
        self.J = len(self.cell_res)
        self.LL = LL_sum
        self.D = dev_stat_sum
        self.Pearson = Pearson_stat_sum
        self.pseudo_R2 = 1 - self.D / dev_y_bar_stat_sum
        self.GF = np.sum((cell_res[:, [0]] * (cell_res[:, [1]] - cell_res[:, [2]])**2) / cell_res[:, [2]])

    def save_stdout_results(self, keys=None):
        # Overall results:
        if keys != None:
            for key in keys:
                res_dict[key] = self.key
        else:
            res_dict = {'p_value': self.p_value, 'n': self.n, 'k': self.k, 'J': self.J, 'LL': self.LL, 'D': self.D, 'Pearson': self.Pearson, 'pseudo_R2': self.pseudo_R2, 'GF': self.GF}

        prefix = 'overall'
        save_results_db(res_dict, prefix, self.model, self.claim_type)

        # Cell results:
        prefix = 'grouped'
        save_results_pkl(self.cell_res, prefix, self.model, self.claim_type)

        # Individual results:
        prefix = 'individual'
        save_results_pkl(self.ind_res, prefix, self.model, self.claim_type)



if __name__ == '__main__':
    for model in ('Poisson', 'Logit', 'Probit', 'C-loglog', 'Gamma', 'InvGaussian', 'NB2'):
        for claim_type in ('casco', 'rcd'):
            x = Estimation(model, claim_type)
            x.save_estimation_results()
            y = Stdout(model, claim_type)
            y.save_stdout_results()
