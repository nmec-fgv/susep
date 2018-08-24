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


## Functions for creation of tables:

# Regression results

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
    
        for var2 in {'n', 'k', 'J', 'LL', 'D_scaled', 'Pearson', 'pseudo_R2', 'GF'}:
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
            aux_dict['C' + model[:2] + 'n-k_Chi2'] = st.chi2.isf(0.05, casco_dict[model]['n'] - casco_dict[model]['k'])
            aux_dict['R' + model[:2] + 'n-k_Chi2'] = st.chi2.isf(0.05, rcd_dict[model]['n'] - rcd_dict[model]['k'])
            aux_dict['C' + model[:2] + 'D_scaled/n-k_Chi2'] = casco_dict[model]['D_scaled'] / aux_dict['C' + model[:2] + 'n-k_Chi2']
            aux_dict['R' + model[:2] + 'D_scaled/n-k_Chi2'] = rcd_dict[model]['D_scaled'] / aux_dict['R' + model[:2] + 'n-k_Chi2']
        else:
            aux_dict['C' + model[:2] + 'J'] = casco_dict[model]['J']
            aux_dict['C' + model[:2] + 'k'] = casco_dict[model]['k']
            aux_dict['R' + model[:2] + 'J'] = rcd_dict[model]['J']
            aux_dict['R' + model[:2] + 'k'] = rcd_dict[model]['k']
            aux_dict['C' + model[:2] + 'Pearson/J-k'] = casco_dict[model]['Pearson'] / (casco_dict[model]['J'] - casco_dict[model]['k'])
            aux_dict['R' + model[:2] + 'Pearson/J-k'] = rcd_dict[model]['Pearson'] / (rcd_dict[model]['J'] - rcd_dict[model]['k'])
            aux_dict['C' + model[:2] + 'J-k_Chi2'] = st.chi2.isf(0.05, casco_dict[model]['J'] - casco_dict[model]['k'])
            aux_dict['R' + model[:2] + 'J-k_Chi2'] = st.chi2.isf(0.05, rcd_dict[model]['J'] - rcd_dict[model]['k'])
            aux_dict['C' + model[:2] + 'D_scaled/J-k_Chi2'] = casco_dict[model]['D_scaled'] / aux_dict['C' + model[:2] + 'J-k_Chi2']
            aux_dict['R' + model[:2] + 'D_scaled/J-k_Chi2'] = rcd_dict[model]['D_scaled'] / aux_dict['R' + model[:2] + 'J-k_Chi2']
    
        aux_dict['C' + model[:2] + 'J-1_Chi2'] = st.chi2.isf(0.05, casco_dict[model]['J'] - 1)
        aux_dict['R' + model[:2] + 'J-1_Chi2'] = st.chi2.isf(0.05, rcd_dict[model]['J'] - 1)
    
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


# Display of factors and levels

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


# Analysis of deviance

def deviance_tables(model, claim_type):
    db_file = data_dir + 'overall_results_' + claim_type + '.db'
    db = shelve.open(db_file)
    aux_dict = {}
    if model == 'LNormal':
        aux_dict['model'] = 'Log-normal'
    elif model == 'InvGaussian':
        aux_dict['model'] = 'Inverse Gaussian'
    else:
        aux_dict['model'] = model

    if claim_type == 'casco':
        aux_dict['claim_type'] = 'own vehicle damage'
    elif claim_type == 'rcd':
        aux_dict['claim_type'] = 'third party liability'

    for int_list in [(('veh_age', 'region'),), (('veh_age', 'sex'),), (('veh_age', 'bonus'),), (('veh_age', 'age'),), (('veh_age', 'cov'),), (('region', 'sex'),), (('region', 'bonus'),), (('region', 'age'),), (('region', 'cov'),), (('sex', 'bonus'),), (('sex', 'age'),), (('sex', 'cov'),), (('bonus', 'age'),), (('bonus', 'cov'),), (('age', 'cov'),), (('veh_age', 'region'), ('veh_age', 'sex'), ('veh_age', 'bonus'), ('veh_age', 'age'), ('veh_age', 'cov'), ('region', 'sex'), ('region', 'bonus'), ('region', 'age'), ('region', 'cov'), ('sex', 'bonus'), ('sex', 'age'), ('sex', 'cov'), ('bonus', 'age'), ('bonus', 'cov'), ('age', 'cov'))]:
        aux_dict[str(int_list) + '_2LLdiff'] = 2 * (db[model + str(int_list)]['LL'] - db[model]['LL'])
        aux_dict[str(int_list) + '_k+'] = db[model + str(int_list)]['k'] - db[model]['k']
        aux_dict[str(int_list) + '_chi2'] = aux_dict[str(int_list) + '_2LLdiff'] / st.chi2.isf(0.05, db[model + str(int_list)]['k'] - db[model]['k'])
        if len(int_list) > 1:
            aux_dict['all_2LLdiff'] = aux_dict.pop(str(int_list) + '_2LLdiff')
            aux_dict['all_k+'] = aux_dict.pop(str(int_list) + '_k+')
            aux_dict['all_chi2'] = aux_dict.pop(str(int_list) + '_chi2')
    
    db.close()
    try:
        with open(tables_dir + 'deviance_table_template.tex', 'r') as cfile:
            template = cfile.read()
    except:
        print('Template file missing')
    
    table = template % aux_dict
    table = table.replace('@', '%')
    table = table.replace(':::model', aux_dict['model'])
    table = table.replace(':::claim_type', aux_dict['claim_type'])
    table = table.replace(':::std_claim_type', claim_type)
    try:
        os.remove(tables_dir + 'deviance_table_' + model + '_' + claim_type + '.tex')
    except OSError:
        pass

    with open(tables_dir + 'deviance_table_' + model + '_' + claim_type + '.tex', 'w') as cfile:
        cfile.write(table)

if __name__ == '__main__':
#    regression_tables()
#    factors_table()
    for model in ('Logit', 'Probit', 'C-loglog', 'LNormal', 'Gamma', 'InvGaussian', 'Poisson'):#, 'NB2'):
        for claim_type in ('casco', 'rcd'):
            deviance_tables(model, claim_type)
