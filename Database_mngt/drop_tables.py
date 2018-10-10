####################################
## Código para delecao de tabelas ## 
####################################

import os
import psycopg2
import pdb

conn = psycopg2.connect("dbname=susep user=ricardob")
cur = conn.cursor()

tables = ('08B', '09A', '09B', '10A', '10B', '11A', '11B', '12A', '12B', '13A', '13B', '14A', '14B', '15A', '15B', '16A', '16B')
names = ('r_auto_', 'r_apolice_', 'r_endosso_', 'r_endosso2_', 'rs_data_', 's_auto_', 's_auto2_')

sql_code = r'''
DROP TABLE IF EXISTS :name:xxx;
'''

for tab in tables:
    for name in names:
        sql = sql_code.replace(':name', name).replace(':xxx', tab)
        cur.execute(sql)


conn.commit()
cur.close()
conn.close()
