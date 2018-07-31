##################################################################
## Generates latex code for table creation from regression data ##
##################################################################

import os
import shelve
import pickle
import pdb
import numpy as np
import scipy.stats as st 

# Data directory:
data_dir = 'persistent/'

# Tables directory:
tables_dir = 'tables/'

def regression_tables():
    db_file = data_dir + 'overall_results_casco.db'
    db = shelve.open(db_file)
    casco_dict = {}
    for key in db.keys():
        casco_dict[key] = db[key]
    
    db.close()
    db_file = data_dir + 'overall_results_rcd.db'
    db = shelve.open(db_file)
    rcd_dict = {}
    for key in db.keys():
        rcd_dict[key] = db[key]
    
    db.close()
    aux_dict = {}
    for model in {'Poisson', 'NB2', 'Logit', 'Probit', 'C-loglog', 'LNormal', 'Gamma', 'InvGaussian'}:
        for var in {'beta', 'std'}:
            for i, item in enumerate(casco_dict[model][var]):
                aux_dict['C' + model[:2] + var + str(i)] = item[0]
    
            for i, item in enumerate(rcd_dict[model][var]):
                aux_dict['R' + model[:2] + var + str(i)] = item[0]
    
        for var2 in {'n', 'k', 'J', 'LL', 'D', 'Pearson', 'pseudo_R2', 'GF'}:
            if var2 == 'LL':
                aux_dict['C' + model[:2] + var2] = - casco_dict[model][var2]
                aux_dict['R' + model[:2] + var2] = - rcd_dict[model][var2]
            else:
                aux_dict['C' + model[:2] + var2] = casco_dict[model][var2]
                aux_dict['R' + model[:2] + var2] = rcd_dict[model][var2]
    
        if model not in {'Logit', 'Probit', 'C-loglog'}:
            aux_dict['C' + model[:2] + 'n'] = casco_dict[model]['n']
            aux_dict['C' + model[:2] + 'k'] = casco_dict[model]['k']
            aux_dict['R' + model[:2] + 'n'] = rcd_dict[model]['n']
            aux_dict['R' + model[:2] + 'k'] = rcd_dict[model]['k']
            aux_dict['C' + model[:2] + 'Pearson/n-k'] = casco_dict[model]['Pearson'] / (casco_dict[model]['n'] - casco_dict[model]['k'])
            aux_dict['R' + model[:2] + 'Pearson/n-k'] = rcd_dict[model]['Pearson'] / (rcd_dict[model]['n'] - rcd_dict[model]['k'])
            aux_dict['C' + model[:2] + 'n-k_Chi2'] = st.chi2.isf(0.95, casco_dict[model]['n'] - casco_dict[model]['k'])
            aux_dict['R' + model[:2] + 'n-k_Chi2'] = st.chi2.isf(0.95, rcd_dict[model]['n'] - rcd_dict[model]['k'])
            aux_dict['C' + model[:2] + 'D/n-k_Chi2'] = casco_dict[model]['D'] / aux_dict['C' + model[:2] + 'n-k_Chi2']
            aux_dict['R' + model[:2] + 'D/n-k_Chi2'] = rcd_dict[model]['D'] / aux_dict['R' + model[:2] + 'n-k_Chi2']
        else:
            aux_dict['C' + model[:2] + 'J'] = casco_dict[model]['J']
            aux_dict['C' + model[:2] + 'k'] = casco_dict[model]['k']
            aux_dict['R' + model[:2] + 'J'] = rcd_dict[model]['J']
            aux_dict['R' + model[:2] + 'k'] = rcd_dict[model]['k']
            aux_dict['C' + model[:2] + 'Pearson/J-k'] = casco_dict[model]['Pearson'] / (casco_dict[model]['J'] - casco_dict[model]['k'])
            aux_dict['R' + model[:2] + 'Pearson/J-k'] = rcd_dict[model]['Pearson'] / (rcd_dict[model]['J'] - rcd_dict[model]['k'])
            aux_dict['C' + model[:2] + 'J-k_Chi2'] = st.chi2.isf(0.95, casco_dict[model]['J'] - casco_dict[model]['k'])
            aux_dict['R' + model[:2] + 'J-k_Chi2'] = st.chi2.isf(0.95, rcd_dict[model]['J'] - rcd_dict[model]['k'])
            aux_dict['C' + model[:2] + 'D/J-k_Chi2'] = casco_dict[model]['D'] / aux_dict['C' + model[:2] + 'J-k_Chi2']
            aux_dict['R' + model[:2] + 'D/J-k_Chi2'] = rcd_dict[model]['D'] / aux_dict['R' + model[:2] + 'J-k_Chi2']
    
        aux_dict['C' + model[:2] + 'J-1_Chi2'] = st.chi2.isf(0.95, casco_dict[model]['J'] - 1)
        aux_dict['R' + model[:2] + 'J-1_Chi2'] = st.chi2.isf(0.95, rcd_dict[model]['J'] - 1)
    
    for aux in ('freq', 'sev', 'freq_bin'):
        for aux2 in ('coeffs', 'diags'):
            try:
                with open(tables_dir + aux + '_' + aux2 + '_table_template.tex', 'r') as cfile:
                    template = cfile.read()
            except:
                print('Template file missing')
            
            table = template % aux_dict
            table = table.replace('@', '%')
            try:
                os.remove(tables_dir + aux + '_' + aux2 + '_table.tex')
            except OSError:
                pass
        
            with open(tables_dir + aux + '_' + aux2 + '_table.tex', 'w') as cfile:
                cfile.write(table)

def factors_table():
    levels_structure = [(0, 1, 2), (3, 4, 5, 6, 7), (8,), (9,), (10, 11, 12, 13), (14, 15, 16), (17, 18, 19)]
    aux_dict = {}
    for claim_type in {'casco', 'rcd'}:
        freq_mat = pickle.load(open(data_dir + 'freq_' + claim_type + '_matrix.pkl', 'rb'))
        sev_mat = pickle.load(open(data_dir + 'sev_' + claim_type + '_matrix.pkl', 'rb'))
        for i, item in enumerate(levels_structure):
            index_b_f = np.where((freq_mat[:, item[0]+3:item[-1]+4]==np.zeros(len(item))).all(-1))[0]
            index_b_s = np.where((sev_mat[:, item[0]+2:item[-1]+3]==np.zeros(len(item))).all(-1))[0]
            aux_dict[claim_type[0].upper()+'b'+str(i)+'e'] = np.sum(freq_mat[:, [1]][index_b_f])
            aux_dict[claim_type[0].upper()+'b'+str(i)+'f'] = 100 * np.sum(freq_mat[:, [0]][index_b_f]) / np.sum(freq_mat[:, [1]][index_b_f])
            aux_dict[claim_type[0].upper()+'b'+str(i)+'as'] = np.average(sev_mat[:, [0]][index_b_s])
            aux_dict[claim_type[0].upper()+'b'+str(i)+'ss'] = np.std(sev_mat[:, [0]][index_b_s])
            for j in item:
                index_f = np.where(freq_mat[:, [j+3]] == 1)[0]
                index_s = np.where(sev_mat[:, [j+2]] == 1)[0]
                aux_dict[claim_type[0].upper()+str(j)+'e'] = np.sum(freq_mat[:, [1]][index_f])
                aux_dict[claim_type[0].upper()+str(j)+'f'] = 100 * np.sum(freq_mat[:, [0]][index_f]) / np.sum(freq_mat[:, [1]][index_f])
                aux_dict[claim_type[0].upper()+str(j)+'as'] = np.average(sev_mat[:, [0]][index_s])
                aux_dict[claim_type[0].upper()+str(j)+'ss'] = np.std(sev_mat[:, [0]][index_s])

    try:
        with open(tables_dir + 'factors_table_template.tex', 'r') as cfile:
            template = cfile.read()
    except:
        print('Template file missing')
    
    table = template % aux_dict
    table = table.replace('e+0', 'e')
    try:
        os.remove(tables_dir + 'factors_table.tex')
    except OSError:
        pass

    with open(tables_dir + 'factors_table.tex', 'w') as cfile:
        cfile.write(table)

    try:
        with open(tables_dir + 'desc_stats_table_template.tex', 'r') as cfile:
            template = cfile.read()
    except:
        print('Template file missing')
    
    table = template % aux_dict
    table = table.replace('e+0', 'e')
    try:
        os.remove(tables_dir + 'desc_stats_table.tex')
    except OSError:
        pass

    with open(tables_dir + 'desc_stats_table.tex', 'w') as cfile:
        cfile.write(table)


if __name__ == '__main__':
    regression_tables()
    factors_table()
