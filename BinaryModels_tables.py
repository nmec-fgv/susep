## Generates latex code for table creation from regression data ##

import os
import shelve
import pdb

try:
    with open('BinaryModels_template.tex', 'r') as cfile:
        template = cfile.read()
    with open('begin_template.tex', 'r') as cfile:
        begin = cfile.read()
except:
    print('At least one tex template file is missing')

try:
    os.remove('BinaryModels_tables.tex')
except OSError:
    pass

with open('BinaryModels_tables.tex', 'w') as cfile:
    cfile.write(begin)

#periods = ('1tr', '2tr', '3tr', '4tr')
periods = ('1tr',)
years = ('08', '09', '10', '11')
dtypes =(('cas', 'count'), ('rcd', 'count'))
distributions = ('Logit', 'Probit', 'Poisson')
for period in periods:
    for aa in years:
        for dtype in dtypes:
            db_file = '/home/pgsqldata/Susep/PoissonResults_' + dtype[0] + '.db'
            db = shelve.open(db_file)
            res_dict = db[period+aa] 
            db.close()
            aux_dict = {}
            aux_dict['period'] = period+aa
            aux_dict['dtype'] = dtype[0]
            for i in range(1, len(res_dict['coeffs'])+1):
                aux_dict['coeffPs'+str(i)] = res_dict['coeffs'][i-1]
                if res_dict['std_MLH'][i-1] != 0:
                    aux_dict['zPs'+str(i)] = res_dict['coeffs'][i-1] / res_dict['std_MLH'][i-1]
                else:
                    aux_dict['zPs'+str(i)] = 0
            for distribution in distributions:
                db_file = '/home/pgsqldata/Susep/BinaryModelResults_' + distribution + '_' + dtype[0] + '.db'
                db = shelve.open(db_file)
                res_dict = db[period+aa] 
                db.close()
                for i in range(1, len(res_dict['coeffs'])+1):
                    if distribution == 'Logit':
                        aux_dict['coeffLt'+str(i)] = res_dict['coeffs'][i-1]
                        if res_dict['std_ML'][i-1] != 0:
                            aux_dict['zLt'+str(i)] = res_dict['coeffs'][i-1] / res_dict['std_ML'][i-1]
                        else:
                            aux_dict['zLt'+str(i)] = 0
                    elif distribution == 'Probit':
                        aux_dict['coeffPt'+str(i)] = res_dict['coeffs'][i-1]
                        if res_dict['std_ML'][i-1] != 0:
                            aux_dict['zPt'+str(i)] = res_dict['coeffs'][i-1] / res_dict['std_ML'][i-1]
                        else:
                            aux_dict['zPt'+str(i)] = 0
                    if distribution == 'Poisson':
                        aux_dict['coeffBP'+str(i)] = res_dict['coeffs'][i-1]
                        if res_dict['std_ML'][i-1] != 0:
                            aux_dict['zBP'+str(i)] = res_dict['coeffs'][i-1] / res_dict['std_ML'][i-1]
                        else:
                            aux_dict['zBP'+str(i)] = 0
                
                
            table = template % aux_dict
    
            with open('BinaryModels_tables.tex', 'a') as cfile:
                cfile.write(table)

with open('BinaryModels_tables.tex', 'a') as cfile:
    cfile.write('\n\\end{document}')
