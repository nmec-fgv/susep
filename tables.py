## Generates latex code for table creation from regression data ##

import os
import shelve
import pdb
import scipy.stats as st 

db_file = 'persistent/overall_results_casco.db'
db = shelve.open(db_file)
casco_dict = {}
for key in db.keys():
    casco_dict[key] = db[key]

db.close()
db_file = 'persistent/overall_results_rcd.db'
db = shelve.open(db_file)
rcd_dict = {}
for key in db.keys():
    rcd_dict[key] = db[key]

db.close()
aux_dict = {}
for model in {'Poisson', 'NB2', 'Logit', 'Probit', 'C-loglog', 'Gamma', 'InvGaussian'}:
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
        aux_dict['C' + model[:2] + 'n-k'] = casco_dict[model]['n'] - casco_dict[model]['k']
        aux_dict['R' + model[:2] + 'n-k'] = rcd_dict[model]['n'] - rcd_dict[model]['k']
        aux_dict['C' + model[:2] + 'Pearson/n-k'] = casco_dict[model]['Pearson'] / (casco_dict[model]['n'] - casco_dict[model]['k'])
        aux_dict['R' + model[:2] + 'Pearson/n-k'] = rcd_dict[model]['Pearson'] / (rcd_dict[model]['n'] - rcd_dict[model]['k'])
        aux_dict['C' + model[:2] + 'n-k_Chi2'] = st.chi2.isf(0.95, casco_dict[model]['n'] - casco_dict[model]['k'])
        aux_dict['R' + model[:2] + 'n-k_Chi2'] = st.chi2.isf(0.95, rcd_dict[model]['n'] - rcd_dict[model]['k'])
        aux_dict['C' + model[:2] + 'D/n-k_Chi2'] = casco_dict[model]['D'] / aux_dict['C' + model[:2] + 'n-k_Chi2']
        aux_dict['R' + model[:2] + 'D/n-k_Chi2'] = rcd_dict[model]['D'] / aux_dict['R' + model[:2] + 'n-k_Chi2']
    else:
        aux_dict['C' + model[:2] + 'J-k'] = casco_dict[model]['J'] - casco_dict[model]['k']
        aux_dict['R' + model[:2] + 'J-k'] = rcd_dict[model]['J'] - rcd_dict[model]['k']
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
            with open(aux + '_' + aux2 + '_table_template.tex', 'r') as cfile:
                template = cfile.read()
        except:
            print('Template file missing')
        
        table = template % aux_dict
        try:
            os.remove(aux + '_' + aux2 + '_table.tex')
        except OSError:
            pass
    
        with open(aux + '_' + aux2 + '_table.tex', 'w') as cfile:
            cfile.write(table)
