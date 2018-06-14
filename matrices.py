###############################################################################
## Loads from data.py and transforms data to make it suitable to frequency   ##
## and severity models of categorical covariates, according to coverage type ##
###############################################################################


import os
import pickle
import numpy as np
import pdb


# Data directories:

data_dir = '/home/pgsqldata/Susep/'
data_dir2 = 'persistent/'


# IPCA (cpi) for adjustment of factor levels expressed in monetary values:

dates = ['2008-01', '2008-02', '2008-03', '2008-04', '2008-05', '2008-06', '2008-07', '2008-08', '2008-09', '2008-10', '2008-11', '2008-12', '2009-01', '2009-02', '2009-03', '2009-04', '2009-05', '2009-06', '2009-07', '2009-08', '2009-09', '2009-10', '2009-11', '2009-12', '2010-01', '2010-02', '2010-03', '2010-04', '2010-05', '2010-06', '2010-07', '2010-08', '2010-09', '2010-10', '2010-11', '2010-12', '2011-01', '2011-02', '2011-03', '2011-04', '2011-05', '2011-06', '2011-07', '2011-08', '2011-09', '2011-10', '2011-11', '2011-12', '2012-01', '2012-02', '2012-03', '2012-04', '2012-05', '2012-06', '2012-07', '2012-08', '2012-09', '2012-10', '2012-11', '2012-12', '2013-01', '2013-02', '2013-03', '2013-04', '2013-05', '2013-06', '2013-07', '2013-08', '2013-09', '2013-10', '2013-11', '2013-12', '2014-01', '2014-02', '2014-03', '2014-04', '2014-05', '2014-06', '2014-07', '2014-08', '2014-09', '2014-10', '2014-11', '2014-12', '2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06', '2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12', '2016-01', '2016-02', '2016-03', '2016-04', '2016-05', '2016-06', '2016-07', '2016-08', '2016-09', '2016-10', '2016-11', '2016-12']

factors = [1.00000000, 1.00540000, 1.01032646, 1.01517603, 1.02075950, 1.02882350, 1.03643679, 1.04192990, 1.04484731, 1.04756391, 1.05227795, 1.05606615, 1.05902313, 1.06410645, 1.06995903, 1.07209895, 1.07724502, 1.08230808, 1.08620438, 1.08881127, 1.09044449, 1.09306156, 1.09612213, 1.10061623, 1.10468851, 1.11297368, 1.12165487, 1.12748748, 1.13391415, 1.13878999, 1.13878999, 1.13890386, 1.13935943, 1.14448654, 1.15307019, 1.16264067, 1.16996531, 1.17967602, 1.18911343, 1.19850743, 1.20773593, 1.21341229, 1.21523241, 1.21717678, 1.22168034, 1.22815524, 1.23343631, 1.23985018, 1.24604943, 1.25302731, 1.25866593, 1.26130913, 1.26938151, 1.27395128, 1.27497044, 1.28045281, 1.28570267, 1.29303118, 1.30066006, 1.30846402, 1.31880089, 1.33014257, 1.33812343, 1.34441261, 1.35180688, 1.35680856, 1.36033627, 1.36074437, 1.36401015, 1.36878419, 1.37658626, 1.38401983, 1.39675281, 1.40443495, 1.41412555, 1.42713550, 1.43669731, 1.44330612, 1.44907934, 1.44922425, 1.45284731, 1.46112854, 1.46726528, 1.47474834, 1.48625137, 1.50468089, 1.52303800, 1.54314210, 1.55409841, 1.56559874, 1.57796697, 1.58775036, 1.59124341, 1.59983613, 1.61295478, 1.62924563, 1.64488638, 1.66577644, 1.68076843, 1.68799573, 1.69829251, 1.71153919, 1.71752957, 1.72646073, 1.73405716, 1.73544440, 1.73995656, 1.74308848]

cpi = dict(zip(dates, factors))


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

def data(data_dict):
    ''' 
    Data preparation for subsequently running models
    '''

    months = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')
    years = ('08', '09', '10', '11')
    for aa in years:
        for mmm in months:
            filename = 'data_' + mmm + aa + '.pkl'
            data = file_load(filename)
            data = {k: v for k, v in data.items() if k in set(('freq_' + data_dict['dependent'], 'sev_' + data_dict['dependent'])) | set(data_dict['factors'].keys()) | set((data_dict['weight'],))}
            aux_index = str(months.index(mmm) + 1)
            if len(aux_index) == 1:
                aux_index = '0' + aux_index

            cpi_index = '20' + aa + '-' + aux_index
            for factor, levels in data_dict['factors'].items():
                for level in levels:
                    aux_arr = np.zeros(len(data['freq_'+data_dict['dependent']]))
                    if data[factor].dtype == 'float':
                        if factor == 'age':
                            aux_arr[np.where(np.logical_and(data[factor] >= level[0], data[factor] < level[1]))] = 1
                        else:
                            aux_arr[np.where(np.logical_and(data[factor] >= level[0] * cpi[cpi_index], data[factor] < level[1] * cpi[cpi_index]))] = 1
                    else:
                        for i in level:
                            aux_arr[np.where(data[factor] == i)] = 1
    
                    try:
                        X = np.hstack((X, aux_arr[:, np.newaxis]))
                    except:
                        X = aux_arr[:, np.newaxis]

            freq_rows = np.unique(X, axis=0)
            if mmm == 'jan' and aa == '08':
                freq_matrix = {}
                sev_matrix = {}

            aux_fmat = np.empty([len(freq_rows), 2 + np.shape(X)[1]])
            for i, row in enumerate(freq_rows):
                index = np.where((X==row).all(-1))[0]
                aux_fmat[i, [0]] = sum(data['freq_'+data_dict['dependent']][index])
                aux_fmat[i, [1]] = sum(data['exposure'][index])
                aux_fmat[i, 2:] = row

            aux_smat = np.hstack((data['sev_'+data_dict['dependent']][:, np.newaxis], data['freq_'+data_dict['dependent']][:, np.newaxis], X))
            aux_smat = aux_smat[np.where(aux_smat[:, [0]]>0)[0]]

            if mmm == 'jan':
                freq_matrix[aa]  = aux_fmat
                sev_matrix[aa] = aux_smat
            else:
                for row in aux_fmat:
                    index2 = np.where((freq_matrix[aa][:, 2:]==row[2:]).all(-1))[0]
                    if len(index2) != 0:
                        freq_matrix[aa][index2[0]][0] += row[0] 
                        freq_matrix[aa][index2[0]][1] += row[1] 
                    else:
                        freq_matrix[aa] = np.vstack((freq_matrix[aa], row))

                sev_matrix[aa] = np.vstack((sev_matrix[aa], aux_smat))


            if  mmm == 'dez':
                aux_farr = np.zeros((len(freq_matrix[aa]), 3))
                aux_sarr = np.zeros((len(sev_matrix[aa]), 3))
                if aa != '08':
                    period = int(aa) - 9
                    aux_farr[:, [period]] = np.ones(len(aux_farr))[:, np.newaxis]
                    aux_sarr[:, [period]] = np.ones(len(aux_sarr))[:, np.newaxis]

                freq_matrix[aa] = np.hstack((freq_matrix[aa], aux_farr))
                sev_matrix[aa] = np.hstack((sev_matrix[aa], aux_sarr))

            print('Frequency and severity matrices loaded w/ ' + data_dict['dependent'] + mmm + aa + ' data')

    freq_matrix = np.vstack((freq_matrix['08'], freq_matrix['09'], freq_matrix['10'], freq_matrix['11']))
    sev_matrix = np.vstack((sev_matrix['08'], sev_matrix['09'], sev_matrix['10'], sev_matrix['11']))
    freq_matrix = np.insert(freq_matrix, 2, 1, axis=1)
    sev_matrix = np.insert(sev_matrix, 1, 1, axis=1)
    try:
        os.remove(data_dir2 + 'freq_' + data_dict['dependent'] + '_matrix.pkl')
    except OSError:
        pass

    with open(data_dir2 + 'freq_' + data_dict['dependent'] + '_matrix.pkl', 'wb') as filename:
        pickle.dump(freq_matrix, filename)

    print('Frequency matrix made persistent in file')
    try:
        os.remove(data_dir2 + 'sev_' + data_dict['dependent'] + '_matrix.pkl')
    except OSError:
        pass

    with open(data_dir2 + 'sev_' + data_dict['dependent'] + '_matrix.pkl', 'wb') as filename:
        pickle.dump(sev_matrix, filename)

    print('Severity matrix made persistent in file')



## Factors and levels determination:

# discrete levels
pol_type_levels = [[i] for i in range(2) if i != 0]
veh_age_levels = [range(3,6), range(6,10), range(10,100)] # base-level = range(0,3)
veh_type_levels = [[i] for i in range(14) if i not in {0, 8, 9}]
region_levels = [range(0,8), range(17,19), list(range(13,17)) + [19], range(20,35), range(35,41)] # base-level = SP (8-12), levels correspond to Sul, RJ, demais Sudeste, Norte/Nordeste, Centro
sex_levels = [[i] for i in range(2) if i != 0]
bonus_c_levels = [[0]] # base-level = 1-9
deduct_type_levels = [[i] for i in range(5) if i != 1]

# continuous levels
age_levels = [(0, .28), (.28, .38), (.50, .60), (.60, 1.50)] # base-level = (.38, .50)
deduct_levels = [(0, .5), (.5, 1), (1.5, 2), (2, 1e5)] # base-level = (1, 1.5)
cov_casco_levels = [(0, 20), (30, 45), (45, 1e5)] # base-level = (20, 30)
cov_rcd_levels = [(0, 60), (60, 80), (120, 1e5)] # base-level = (80, 120)
cov_app_levels = [(0, 0.01), (0.01, 10), (30, 60), (60, 1e5)] # base-level = (10, 30)

# factors
factors = {'veh_age': veh_age_levels, 'region': region_levels, 'sex': sex_levels, 'bonus_c': bonus_c_levels, 'age': age_levels, 'cov_casco': cov_casco_levels}
data_dict = {'dependent': 'casco', 'factors': factors, 'weight': 'exposure'}

##

if __name__ == '__main__':
    x = data(data_dict) 
