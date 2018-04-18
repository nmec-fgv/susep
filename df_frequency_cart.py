############################################################
## Captura de dados sobre frequencia das tabelas rs_mmmaa ##
############################################################

import os
import pickle
import psycopg2

conn = psycopg2.connect("dbname=susep user=ricardob")
cur = conn.cursor()

years = ('06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16')
months = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')

sql_code = '''SELECT inicio_vig, fim_vig, sinistro, endosso FROM rs_:mmm:aa WHERE cod_tarif in (' 10', ' 11', '14A', '14B', '14C', ' 15', ' 16', ' 17', ' 18', ' 19', ' 20', ' 21', ' 22', ' 23');'''

for aa in years:
    for mmm in months:
        sql_code2 = sql_code.replace(':aa', aa)
        sql_code2 = sql_code2.replace(':mmm', mmm)
        print(sql_code2)
        cur.execute(sql_code2)
        data = cur.fetchall()

        filename = 'freq_dat_' + mmm + aa + '_cart.pkl'
        try:
            os.remove('Data/' + filename)
        except OSError:
            pass

        with open('Data/' + filename, 'wb') as file:
            pickle.dump(data, file)

conn.commit()
cur.close()
conn.close()
