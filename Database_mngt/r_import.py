########################################################
## Código para inserção de valores nas tabelas r_auto ##
########################################################

import os
import psycopg2
import pdb

conn = psycopg2.connect("dbname=susep user=ricardob")
cur = conn.cursor()

tables = ('08B', '09A', '09B', '10A', '10B', '11A', '11B', '12A', '12B', '13A', '13B', '14A', '14B', '15A', '15B', '16A', '16B')

sql_code = r'''
DROP TABLE IF EXISTS r_auto_:xxx;

CREATE TABLE r_auto_:xxx (
    cod_apo         varchar(20),
    endosso         varchar(10),
    cod_end         varchar(10),
    item            varchar(10),
    tipo_pes        varchar(10),
    modalidade      varchar(10),
    tipo_prod       varchar(10),
    cobertura       varchar(10),
    cod_modelo      varchar(10),
    ano_modelo      integer,
    cod_tarif       varchar(10),
    regiao          varchar(10),
    cod_cont        varchar(10),
    tipo_franq      varchar(10),
    val_franq       real,
    perc_fator      real,
    tab_ref         varchar(10),
    is_casco        real,
    is_rcdmat       real,
    is_rcdc         real,
    is_rcdmor       real,
    is_app_ma       real,
    is_app_ipa      real,
    is_app_dmh      real,
    pre_casco       real,
    pre_casco_co    real,
    pre_rcdmat      real,
    pre_rcdc        real,
    pre_rcdmor      real,
    pre_app_ma      real,
    pre_app_ia      real,
    pre_app_dm      real,
    pre_outros      real,
    inicio_vig      date,
    fim_vig         date,
    perc_bonus      real,
    clas_bonus      varchar(10),
    perc_corr       real,
    sexo            varchar(10),
    data_nasc       date,
    tempo_hab       integer,
    utilizacao      varchar(10),
    cep_util        integer,
    cep_per         integer,
    sinal           varchar(10)
);
'''

sql_code2 = r'''
DROP TABLE IF EXISTS r_auto_:xxx;

CREATE TABLE r_auto_:xxx (
    cod_apo         varchar(20),
    endosso         varchar(10),
    cod_end         varchar(10),
    item            varchar(10),
    tipo_pes        varchar(10),
    modalidade      varchar(10),
    tipo_prod       varchar(10),
    cobertura       varchar(10),
    cod_modelo      varchar(10),
    ano_modelo      integer,
    cod_tarif       varchar(10),
    regiao          varchar(10),
    cod_cont        varchar(10),
    tipo_franq      varchar(10),
    val_franq       real,
    perc_fator      real,
    tab_ref         varchar(10),
    is_casco        real,
    is_rcdmat       real,
    is_rcdc         real,
    is_rcdmor       real,
    is_app_ma       real,
    is_app_ipa      real,
    is_app_dmh      real,
    pre_casco       real,
    pre_casco_co    real,
    pre_rcdmat      real,
    pre_rcdc        real,
    pre_rcdmor      real,
    pre_app_ma      real,
    pre_app_ia      real,
    pre_app_dm      real,
    pre_outros      real,
    inicio_vig      date,
    fim_vig         date,
    perc_bonus      real,
    clas_bonus      varchar(10),
    perc_corr       real,
    sexo            varchar(10),
    data_nasc       date,
    tempo_hab       integer,
    utilizacao      varchar(10),
    cep_util        integer,
    cep_per         integer,
    sinal           varchar(10),
    cod_seg_apolice varchar(30),
    id_calculado    varchar(10)
);
'''

sql = {}
for tab in tables:
    if tab == '16A':
        sql['tab'] = sql_code2
    else:
        sql['tab'] = sql_code
    for csv in ('00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'):
        if os.path.isfile('/home/pgsqldata/Susep/' + tab + '/R_AUTO_20' + tab + '-0' + csv + '.csv'):
            sql['tab'] += "\nCOPY r_auto_:xxx from '/home/pgsqldata/Susep/" + tab + "/R_AUTO_20" + tab + "-0" + csv + ".csv' delimiter ';' csv header null 'NULL';"
        else:
            break
    sql['tab'] = sql['tab'].replace(':xxx', tab)
    cur.execute(sql['tab'])
    print('Tabela r_auto_' + tab + ' carregada com sucesso')


conn.commit()
cur.close()
conn.close()
