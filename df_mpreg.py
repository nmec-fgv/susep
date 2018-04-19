#####################################################################
## Captura de dados de características a priori e sobre frequencia ##
## das tabelas rs_mmmaa para regressões de tipo mixed poisson.     ##
#####################################################################

import os
import pickle
import psycopg2

conn = psycopg2.connect("dbname=susep user=ricardob")
cur = conn.cursor()

years = ('08', '09', '10', '11')
months = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')

sql_code = '''SELECT tipo_pes, ano_modelo, cod_tarif, regiao, inicio_vig, fim_vig, sexo, data_nasc, sinistro, endosso FROM rs_:mmm:aa WHERE tipo_pes in ('F') AND ano_modelo > 1900 AND cod_tarif in (' 10', ' 11', '14A', '14B', '14C', ' 15', ' 16', ' 17', ' 18', ' 19', ' 20', ' 21', ' 22', ' 23') AND regiao in ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41') AND sexo in ('F', 'M') AND data_nasc > '1900-01-01';'''

for aa in years:
    for mmm in months:
        sql_code2 = sql_code.replace(':aa', aa)
        sql_code2 = sql_code2.replace(':mmm', mmm)
        print(sql_code2)
        cur.execute(sql_code2)
        data = cur.fetchall()

        filename = 'data_mpregres_' + mmm + aa + '.pkl'
        try:
            os.remove('Data/' + filename)
        except OSError:
            pass

        with open('Data/' + filename, 'wb') as file:
            pickle.dump(data, file)

conn.commit()
cur.close()
conn.close()
