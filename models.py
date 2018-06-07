#########################################################################
## Regression models of counts and claims data for auto insurance cost ##
#########################################################################


import os
import pickle
import shelve
import numpy as np
import sympy
from scipy.special import factorial
from scipy.stats import norm
import pdb


# Lower bound parameters for logs and ratios, plus precision parameter:

lb_log = 1e-323
lb_ratio = 1e-308
prec_param = 1e-4


# Data directory:

data_dir = '/home/pgsqldata/Susep/'


# Auxiliary functions:

def file_load(filename):
    try:
        os.path.exists(data_dir + filename)
        with open(data_dir + filename, 'rb') as file:
            x = pickle.load(file)
    except:
        print('File ' + filename + ' not found')

    return x

def perc_categ(x_array):
    res = [(0, np.percentile(x_array, 20)), (np.percentile(x_array, 20), np.percentile(x_array, 40)), (np.percentile(x_array, 60), np.percentile(x_array, 80)), (np.percentile(x_array, 80), np.percentile(x_array, 100))]
    return res


# Classes:

class Data:
    ''' 
    Data preparation for subsequently running models

    '''

    def __init__(self, period, aa, data_dict):
        periods = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez', '1tr', '2tr', '3tr', '4tr')
        years = ('08', '09', '10', '11')

        if period not in periods and aa not in years:
            raise Exception('period invalid or outside permissible range')

        if period in periods[:12]:
            mmm = period
            filename = 'data_' + mmm + aa + '.pkl'
            data = file_load(filename)
            data = {k: v for k, v in data.items() if k in set((data_dict['dependent'],)) | set(data_dict['factors'].keys()) | data_dict['cont_vars']}

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

            for i in range(3):
                aux[str(i)] = {k: v for k, v in aux[str(i)].items() if k in set((data_dict['dependent'],)) | set(data_dict['factors'].keys()) | data_dict['cont_vars']}
            data = aux['0']
            for key in data.keys():
                for i in {'1', '2'}:
                    data[key] = np.concatenate((data[key], aux[i][key]))

        if data_dict['binary'] == 'no':
            for key in data.keys():
                if key[:4] == 'freq' or key[:3] == 'sev':
                    self.y = data[key]
        elif data_dictionary['binary'] == 'yes':
            for key in data.keys():
                if key[:4] == 'freq':
                    self.y = data[key]
                    self.y[self.y > 1] = 1
                elif key[:3] == 'sev':
                    raise Exception('Only frequency data takes positive binary argument')
        else:
            raise Exception('Invalid value for binary arg of data_dict')

        if data_dict['offset'] == 'log_exposure':
            self.X = np.log(data['exposure'])[:, np.newaxis]

        if hasattr(self, 'X'):
            self.X = np.hstack((self.X, np.ones(len(self.X))[:, np.newaxis]))
        else:
            self.X = np.ones(len(data[data_dict['dependent']]))[:, np.newaxis]

        if bool(data_dict['cont_vars'] - {'exposure'}):
            for key in data_dict['cont_vars'] - {'exposure'}:
                self.X = np.hstack((self.X, data[key][:, np.newaxis]))

        for factor, levels in data_dict['factors'].items():
            if levels == 'percentiles':
                levels = perc_categ(data[factor])
            
            for level in levels:
                aux_arr = np.zeros(len(self.X))
                if data[factor].dtype == 'float':
                    if level != levels[-1]:
                        aux_arr[np.where(np.logical_and(data[factor] >= level[0], data[factor] < level[1]))] = 1
                    else:
                        aux_arr[np.where(np.logical_and(data[factor] >= level[0], data[factor] <= level[1]))] = 1
                else:
                    for i in level:
                        aux_arr[np.where(data[factor] == i)] = 1

                self.X = np.hstack((self.X, aux_arr[:, np.newaxis]))



## Factors and levels determination:

# discrete levels
pol_type_levels = [[i] for i in range(2) if i != 0]
veh_age_levels = [range(3,6), range(6,9), range(9,12), range(12,15), range(15,100)] # base-level = range(0,3)
veh_type_levels = [[i] for i in range(14) if i not in {0, 8, 9}]
region_levels = [[i] for i in range(41) if i != 10]
sex = [[i] for i in range(2) if i != 0]
bonus_c_levels = [[i] for i in range(10) if i != 0]
deduct_type_levels = [[i] for i in range(5) if i != 1]

# continuous levels
age_levels = [(.17, .22), (.22, .25), (.25, .30), (.30, .35), (.35, .40), (.50, .60), (.60, 1.50)] # base-level = (.40, .50)
deduct_levels = [(0, .5), (.5, 1), (1.5, 2), (2, 1e5)] # base-level = (1, 1.5)
cov_casco_levels = [(0, 15), (15, 30), (45, 60), (60, 1e5)] # base-level = (30, 45)
cov_rcd_levels = [(0, 30), (30, 60), (90, 120), (120, 1e5)] # base-level = (60, 90)
cov_app_levels = [(0, 0.01), (0.01, 10), (30, 60), (60, 1e5)] # base-level = (10, 30)

# factors
factors = {'pol_type': pol_type_levels, 'veh_age': veh_age_levels, 'veh_type': veh_type_levels, 'region': region_levels, 'bonus_c': bonus_c_levels, 'deduct_type': deduct_type_levels, 'age': age_levels, 'deduct': deduct_levels, 'cov_casco': cov_casco_levels, 'cov_rcd': cov_rcd_levels, 'cov_app': cov_app_levels}
cont_vars = {'exposure'}

##

if __name__ == '__main__':
    data_dict = {'dependent': 'freq_rcd', 'factors': factors, 'cont_vars': cont_vars, 'binary': 'no', 'offset': 'log_exposure'}
    x = Data('1tr', '08', data_dict) 
    pdb.set_trace()
