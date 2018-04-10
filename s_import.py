########################################################
## Código para inserção de valores nas tabelas s_auto ##
########################################################

import os
import psycopg2

conn = psycopg2.connect("dbname=susep user=ricardob")
cur = conn.cursor()

tables = ('14A', '14B', '15A', '15B', '16A', '16B')

sql_code = r'''
DROP TABLE IF EXISTS s_auto_:xxx;

CREATE TABLE s_auto_:xxx (
    cod_apo         varchar(20),
    endosso         varchar(10),
    item            varchar(06),
    modalidade      varchar(1),
    tipo_prod       varchar(1),
    cobertura       varchar(1),
    cod_modelo      varchar(8),
    ano_modelo      integer,
    cod_tarif       varchar(3),
    regiao          varchar(2),
    cod_cont        varchar(1),
    evento          varchar(1),
    indeniz         real,
    val_salvad      real,
    d_salvado       date,
    val_ress        real,
    d_ress          date,
    d_avi           date,
    d_liq           date,
    d_ocorr         date,
    causa           varchar(1),
    sexo            varchar(1),
    d_nasc          date,
    cep             integer
);
'''

sql_code2 = r'''
DROP TABLE IF EXISTS s_auto_:xxx;

CREATE TABLE s_auto_:xxx (
    cod_apo         varchar(20),
    endosso         varchar(10),
    item            varchar(06),
    modalidade      varchar(1),
    tipo_prod       varchar(1),
    cobertura       varchar(1),
    cod_modelo      varchar(8),
    ano_modelo      integer,
    cod_tarif       varchar(3),
    regiao          varchar(2),
    cod_cont        varchar(1),
    evento          varchar(1),
    indeniz         real,
    val_salvad      real,
    d_salvado       date,
    val_ress        real,
    d_ress          date,
    d_avi           date,
    d_liq           date,
    d_ocorr         date,
    causa           varchar(1),
    sexo            varchar(1),
    d_nasc          date,
    cep             integer,
    cod_seg_apolice varchar(30),
    id_calculado    varchar(10)
);
'''

sql={}
for tab in tables:
    if tab == '16A':
        sql['tab'] = sql_code2
    else:
        sql['tab'] = sql_code
    sql['tab'] += "\nCOPY s_auto_:xxx from '/home/pgsqldata/Susep/" + tab + "/S_AUTO_20" + tab + ".csv' delimiter ';' csv header null 'NULL';"
    sql['tab'] = sql['tab'].replace(':xxx', tab)
    cur.execute(sql['tab'])
    print('Tabela s_auto_' + tab + ' carregada com sucesso')

conn.commit()
cur.close()
conn.close()
