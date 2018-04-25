############################################################
## Código para inserção de valores nas tabelas r_endosso2 ##
############################################################

import os
import psycopg2

conn = psycopg2.connect("dbname=susep user=ricardob")
cur = conn.cursor()

tables = ('08B', '09A', '09B', '10A', '10B', '11A', '11B', '12A', '12B', '13A', '13B', '14A', '14B', '15A', '15B', '16A', '16B')

sql_code = r'''
DROP TABLE IF EXISTS r_endosso2_:xxx;

CREATE TABLE r_endosso2_:xxx
    AS SELECT
        cod_apo,
        cod_modelo,
        json_object_agg(endosso, (cod_end, inicio_vig)) AS endosso
        FROM r_endosso_:xxx
        GROUP BY cod_apo, cod_modelo;
'''

sql={}
for tab in tables:
    sql['tab'] = sql_code
    sql['tab'] = sql['tab'].replace(':xxx', tab)
    cur.execute(sql['tab'])
    print('Tabela r_endosso2_' + tab + ' carregada com sucesso')

conn.commit()
cur.close()
conn.close()
