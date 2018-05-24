## Generates latex code for table creation from regression data ##

import os
import shelve
import pdb

try:
    with open('Poisson_template.tex', 'r') as cfile:
        template = cfile.read()
    with open('begin_template.tex', 'r') as cfile:
        begin = cfile.read()
except:
    print('At least one tex template file is missing')

try:
    os.remove('Poisson_tables.tex')
except OSError:
    pass

with open('Poisson_tables.tex', 'w') as cfile:
    cfile.write(begin)

periods = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez', '1tr', '2tr', '3tr', '4tr')
years = ('08', '09', '10', '11')
dtypes =(('cas', 'count'), ('rcd', 'count'), ('app', 'count'))
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
                aux_dict['coeff'+str(i)] = res_dict['coeffs'][i-1]
                aux_dict['MLH'+str(i)] = res_dict['std_MLH'][i-1]
                aux_dict['MLOP'+str(i)] = res_dict['std_MLOP'][i-1]
                aux_dict['NB1'+str(i)] = res_dict['std_NB1'][i-1]
                aux_dict['RS'+str(i)] = res_dict['std_RS'][i-1]
                if res_dict['std_MLH'][i-1] != 0:
                    aux_dict['tstat'+str(i)] = res_dict['coeffs'][i-1] / res_dict['std_MLH'][i-1]
                else:
                    aux_dict['tstat'+str(i)] = 0
            table = template % aux_dict

            with open('Poisson_tables.tex', 'a') as cfile:
                cfile.write(table)

with open('Poisson_tables.tex', 'a') as cfile:
    cfile.write('\n\\end{document}')
