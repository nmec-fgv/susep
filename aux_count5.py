##############################################
## Auxiliary code for counting observations ##
##############################################

import os
import pickle
import numpy as np
import pdb


# Data directories:

data_dir = '/home/pgsqldata/Susep/'
data_dir2 = 'persistent/'

# Auxiliary functions:

def file_load(filename):
    try:
        os.path.exists(data_dir + filename)
        with open(data_dir + filename, 'rb') as file:
            x = pickle.load(file)
    except:
        print('File ' + filename + ' not found')

    return x

# Main function:

def main():
    ''' 
    Data preparation for subsequently running models
    '''

    months = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')
    years = ('08', '09', '10', '11')
    casco_sum = 0
    rcd_sum = 0
    total_sum = 0
    for aa in years:
        for mmm in months:
            print(mmm+aa)
            filename = 'data_' + mmm + aa + '.pkl'
            data = file_load(filename)
            casco_sum += len(np.where(data['freq_casco']>4)[0])
            rcd_sum += len(np.where(data['freq_rcd']>4)[0])
            total_sum += len(data['freq_casco'])

    print('Casco:', casco_sum/total_sum, 'Rcd:', rcd_sum/total_sum)

if __name__ == '__main__':
    main()
