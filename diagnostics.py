#########################################################################
## Regression models of counts and claims data for auto insurance cost ##
#########################################################################

import os
import sys
import pickle
import shelve
import numpy as np
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Data directories:

data_dir = 'persistent/'
plot_dir = 'plots/'

# Auxiliary functions:

def file_load(filename):
    try:
        os.path.exists(data_dir + filename)
        with open(data_dir + filename, 'rb') as cfile:
            x = pickle.load(cfile)
    except:
        print('File ' + filename + ' not found')

    return x

class Diagnostics:
    def __init__(self, model, claim_type):
        self.model = model
        self.claim_type = claim_type
        if self.model in {'Poisson', 'NB2', 'Logit', 'Probit', 'C-loglog'}:
            self.model_type = 'freq'
        elif self.model in {'LNormal', 'Gamma', 'InvGaussian'}:
            self.model_type = 'sev'

        filename = 'grouped_results_' + model + '_' + claim_type + '.pkl'
        self.cell_res = file_load(filename)

    def plot01(self):
        '''Plots raw residual of grouped data, defined as y_bar - mu, against mu'''
        index = np.where(self.cell_res[:, [0]] > 0)[0]
        x = self.cell_res[:, [2]][index]
        y = self.cell_res[:, [1]][index] - self.cell_res[:, [2]][index]
        color = 1 - (self.cell_res[:, [0]][index] / np.amax(self.cell_res[:, [0]][index])) 
        color = np.hstack((color, color, color))
        color = color**2 * .1
        plt.scatter(x, y, s=1, c=color, alpha=.3)
        plt.savefig(plot_dir + self.model + self.claim_type + 'plot01.png')
        plt.close()


if __name__ == '__main__':
    for model in ('Poisson', 'NB2', 'Logit', 'Probit', 'C-loglog', 'LNormal', 'Gamma', 'InvGaussian'):
        for claim_type in ('casco', 'rcd',):
            x = Diagnostics(model, claim_type)
            x.plot01()
