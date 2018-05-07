##################################################
## Captura de dados da base para arquivo pickle ##
##################################################

import os
import pickle
import psycopg2

conn = psycopg2.connect("dbname=susep user=ricardob")
cur = conn.cursor()

years = ('08', '09', '10', '11')
months = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')

sql_code = '''SELECT cod_cont, ano_modelo, cod_tarif, regiao, inicio_vig, fim_vig, sexo, data_nasc, sinistro, endosso, clas_bonus, tipo_franq, val_franq, is_casco, is_rcdmat, is_rcdc, is_rcdmor, is_app_ma, is_app_ipa, is_app_dmh FROM rs_:mmm:aa WHERE tipo_pes = 'F' AND cobertura = '1' AND cod_cont in ('1', '2') AND ano_modelo > 1900 AND cod_tarif in (' 10', '10 ', ' 11', '11 ', '14A', '14B', '14C', ' 15', '15 ', ' 16', '16 ', ' 17', '17 ', ' 18', '18 ', ' 19', '19 ', ' 20', '20 ', ' 21', '21 ', ' 22', '22 ', ' 23', '23 ') AND regiao in ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41') AND sexo in ('F', 'M') AND data_nasc > '1900-01-01' AND clas_bonus in ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9') AND tipo_franq in ('1', '2', '3', '4', '9') AND val_franq >= 0 AND is_casco >= 0 AND is_rcdmat >= 0 AND is_rcdc >= 0 AND is_rcdmor >= 0 AND is_app_ma >= 0 AND is_app_ipa >= 0 AND is_app_dmh >= 0 AND pre_casco >= 0 AND pre_rcdmat >= 0 AND pre_rcdc >= 0 AND pre_rcdmor >= 0 AND pre_app_ma >= 0 AND pre_app_ia >= 0 AND pre_app_dm >= 0;'''

for aa in years:
    for mmm in months:
        sql_code2 = sql_code.replace(':aa', aa)
        sql_code2 = sql_code2.replace(':mmm', mmm)
        cur.execute(sql_code2)
        data = cur.fetchall()
        filename = 'data_' + mmm + aa + '_raw.pkl'
        try:
            os.remove('/home/pgsqldata/Susep/' + filename)
        except OSError:
            pass

        with open('/home/pgsqldata/Susep/' + filename, 'wb') as file:
            pickle.dump(data, file)

        print('File data_' + mmm + aa + '_raw.pkl saved')

conn.commit()
cur.close()
conn.close()
