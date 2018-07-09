############################################################################
## Creates tables in Postgres and insert values from data_mmmaa.pkl files ##
############################################################################


import os
import pickle
import numpy as np
import psycopg2
import pdb


# Data directories:

data_dir = '/home/pgsqldata/Susep/'


# Auxiliary functions:

def file_load(filename):
    try:
        os.path.exists(data_dir + filename)
        with open(data_dir + filename, 'rb') as file:
            x = pickle.load(file)
    except:
        print('File ' + filename + ' not found')

    return x

sql_code1_begin = r'''
DROP TABLE IF EXISTS datapkl_:mmm:aa;

CREATE TABLE datapkl_:mmm:aa (
'''

sql_code1_end = r'''
);
'''

sql_code2A = r'''
INSERT INTO datapkl_:mmm:aa (
'''

sql_code2B = r'''
) VALUES (
'''

sql_code2C = r'''
);
'''

if __name__ == '__main__':
    years = ('08', '09', '10', '11')
    months = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')
    for aa in years:
        for mmm in months:
            filename = 'data_' + mmm + aa + '.pkl'
            data = file_load(filename)
            aux_str1 = ''
            aux_str2 = ''
            aux_str3 = ''
            aux_data = np.empty((len(data[next(iter(data.keys()))]), len(data)))
            for i, key in enumerate(data.keys()):
                if key[:3] == 'sev':
                    data[key] = np.average(data[key], axis=1)

                aux_str1 += key + ' real, '
                aux_str2 += key + ', '
                aux_str3 += ' %s,'
                aux_data[:, [i]] = data[key][:, np.newaxis]

            aux_str1 = aux_str1[:-2]
            aux_str2 = aux_str2[:-2]
            aux_str3 = aux_str3[:-1]
            sql_code_begin = sql_code1_begin.replace(':aa', aa)
            sql_code_begin = sql_code_begin.replace(':mmm', mmm)
            sql_code = sql_code_begin + aux_str1 + sql_code1_end
            sql_code_begin = sql_code2A.replace(':aa', aa)
            sql_code_begin = sql_code_begin.replace(':mmm', mmm)
            sql_code2 = sql_code_begin + aux_str2 + sql_code2B + aux_str3 + sql_code2C
            conn = psycopg2.connect("dbname=susep user=ricardob")
            cur = conn.cursor()
            cur.execute(sql_code)
            for row in aux_data:
                cur.execute(sql_code2, [i for i in row] )

            conn.commit()
            cur.close()
            conn.close()
            print('Dados ' + mmm + aa + ' carregados com sucesso')
