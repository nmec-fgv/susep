## Generates latex code for table creation from regression data ##

import os
import shelve
import pdb

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
aux_dict = {'percent': '%'}
for model in {'Poisson', 'NB2', 'Gamma', 'InvGaussian'}:
    for var in {'beta', 'std'}:
        for i, item in enumerate(casco_dict[model][var]):
            aux_dict['C' + model[0] + var + str(i)] = item[0]

        for i, item in enumerate(rcd_dict[model][var]):
            aux_dict['R' + model[0] + var + str(i)] = item[0]

    for var2 in {'LL', 'D', 'D_Chi2', 'Pearson', 'pseudo_R2', 'GF', 'GF_Chi2'}:
        if var2 == 'LL':
            aux_dict['C' + model[0] + var2] = - casco_dict[model][var2]
            aux_dict['R' + model[0] + var2] = - rcd_dict[model][var2]
        else:
            aux_dict['C' + model[0] + var2] = casco_dict[model][var2]
            aux_dict['R' + model[0] + var2] = rcd_dict[model][var2]

    aux_dict['C' + model[0] + 'n-k'] = casco_dict[model]['n'] - casco_dict[model]['k']
    aux_dict['R' + model[0] + 'n-k'] = rcd_dict[model]['n'] - rcd_dict[model]['k']
    aux_dict['C' + model[0] + 'Pearson/n-k'] = casco_dict[model]['Pearson'] / (casco_dict[model]['n'] - casco_dict[model]['k'])
    aux_dict['R' + model[0] + 'Pearson/n-k'] = rcd_dict[model]['Pearson'] / (rcd_dict[model]['n'] - rcd_dict[model]['k'])
    aux_dict['C' + model[0] + 'D/D_Chi2'] = casco_dict[model]['D'] / casco_dict[model]['D_Chi2']
    aux_dict['R' + model[0] + 'D/D_Chi2'] = rcd_dict[model]['D'] / rcd_dict[model]['D_Chi2']

for aux in ('freq', 'sev'):
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
