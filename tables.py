## Generates latex code for table creation from regression data ##

import os
import shelve
import pdb

template_file = 'table_template.tex'
product_file = 'table.tex'
try:
    os.remove(product_file)
except OSError:
    pass

db_file = 'persistent/results_casco.db'
db = shelve.open(db_file)
casco_dict = {}
for key in db.keys():
    casco_dict[key] = db[key]

db.close()
db_file = 'persistent/results_rcd.db'
db = shelve.open(db_file)
rcd_dict = {}
for key in db.keys():
    rcd_dict[key] = db[key]

db.close()
aux_dict = {}
for model in {'Poisson', 'Gamma', 'InvGaussian'}:
    for var in {'beta', 'std'}:
        for i, item in enumerate(casco_dict[model][var]):
            aux_dict['C' + model[0] + var + str(i)] = item[0]

        for i, item in enumerate(rcd_dict[model][var]):
            aux_dict['R' + model[0] + var + str(i)] = item[0]
 
try:
    with open(template_file, 'r') as cfile:
        template = cfile.read()
except:
    print('Template file missing')

table = template % aux_dict
with open(product_file, 'a') as cfile:
    cfile.write(table)
