##########################################################################
## Provides function for creation of dummies given desired interactions ##
##########################################################################

import numpy as np
import pdb

factors_list = {'veh_age': (0, 2), 'region': (3, 7), 'sex': (8,), 'bonus': (9,), 'age': (10, 13), 'cov': (14, 16), 'year': (17, 19)} 

def function(X, dependent, interactions_list):
    if dependent == 'freq':
        aux_disp = 3
    elif dependent == 'sev':
        aux_disp = 2

    for item in interactions_list:
        item0f = factors_list[item[0]][0]
        item0l = factors_list[item[0]][-1]
        item0_size = item0l - item0f + 2
        item1f = factors_list[item[1]][0]
        item1l = factors_list[item[1]][-1]
        item1_size = item1l - item1f + 2
        item_size = (item0_size - 1) * (item1_size - 1)
        X_add = np.zeros((np.shape(X)[0], item_size))
        aux_pos = -1
        for i in range(item0_size):
            level_i = item0f + aux_disp + i
            for j in range(item1_size):
                aux_pos += 1
                level_j = item1f + aux_disp + j
                index = np.where((X[:, [level_i, level_j]] == [1, 1]).all(-1))[0] 
                X_add[index, aux_pos] = 1 

        X = np.hstack((X, X_add))
        return X
