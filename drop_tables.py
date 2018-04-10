###########################################################
## Código para deleção de tabelas da base de dados susep ##
###########################################################

import psycopg2

conn = psycopg2.connect("dbname=susep user=ricardob")
cur = conn.cursor()

#prefixes = ('r_apolice_', 'r_endosso2_', 's_auto2_')
prefixes = ('rs_jan', 'rs_fev', 'rs_mar', 'rs_abr', 'rs_mai', 'rs_jun', 'rs_jul', 'rs_ago', 'rs_set', 'rs_out', 'rs_nov', 'rs_dez')
#tables = ('08B', '09A', '09B', '10A', '10B', '11A', '11B', '12A', '12B', '13A', '13B', '14A', '14B', '15A', '15B', '16A', '16B')
tables = ('06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16')

sql_code = r'''
DROP TABLE IF EXISTS :yyy:xxx;
'''

for prefix in prefixes:
    code = sql_code.replace(':yyy', prefix)
    for tab in tables:
        code2 = code.replace(':xxx', tab)
        cur.execute(code2)

conn.commit()
cur.close()
conn.close()
