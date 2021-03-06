########################################################
## Código para inserção de valores nas tabelas rs_data##
########################################################

import psycopg2

conn = psycopg2.connect("dbname=susep user=ricardob")
cur = conn.cursor()

tables = ('08B', '09A', '09B', '10A', '10B', '11A', '11B', '12A', '12B', '13A', '13B', '14A', '14B', '15A', '15B', '16A', '16B')

sql_code = r'''
DROP TABLE IF EXISTS rs_data_:xxx;

CREATE TABLE rs_data_:xxx
AS SELECT
    r_apolice_:xxx.cod_apo,
    r_apolice_:xxx.item,
    r_apolice_:xxx.tipo_pes,
    r_apolice_:xxx.modalidade,
    r_apolice_:xxx.tipo_prod,
    r_apolice_:xxx.cobertura,
    r_apolice_:xxx.cod_modelo,
    r_apolice_:xxx.ano_modelo,
    r_apolice_:xxx.cod_tarif,
    r_apolice_:xxx.regiao,
    r_apolice_:xxx.cod_cont,
    r_apolice_:xxx.tipo_franq,
    r_apolice_:xxx.val_franq,
    r_apolice_:xxx.perc_fator,
    r_apolice_:xxx.tab_ref,
    r_apolice_:xxx.is_casco,
    r_apolice_:xxx.is_rcdmat,
    r_apolice_:xxx.is_rcdc,
    r_apolice_:xxx.is_rcdmor,
    r_apolice_:xxx.is_app_ma,
    r_apolice_:xxx.is_app_ipa,
    r_apolice_:xxx.is_app_dmh,
    r_apolice_:xxx.pre_casco,
    r_apolice_:xxx.pre_casco_co,
    r_apolice_:xxx.pre_rcdmat,
    r_apolice_:xxx.pre_rcdc,
    r_apolice_:xxx.pre_rcdmor,
    r_apolice_:xxx.pre_app_ma,
    r_apolice_:xxx.pre_app_ia,
    r_apolice_:xxx.pre_app_dm,
    r_apolice_:xxx.pre_outros,
    r_apolice_:xxx.inicio_vig,
    r_apolice_:xxx.fim_vig,
    r_apolice_:xxx.perc_bonus,
    r_apolice_:xxx.clas_bonus,
    r_apolice_:xxx.perc_corr,
    r_apolice_:xxx.sexo,
    r_apolice_:xxx.data_nasc,
    r_apolice_:xxx.tempo_hab,
    r_apolice_:xxx.utilizacao,
    r_apolice_:xxx.cep_util,
    r_apolice_:xxx.cep_per,
    s_auto2_:xxx.sinistro,
    r_endosso2_:xxx.endosso
FROM r_apolice_:xxx LEFT OUTER JOIN s_auto2_:xxx ON r_apolice_:xxx.cod_apo = s_auto2_:xxx.cod_apo AND r_apolice_:xxx.cod_modelo = s_auto2_:xxx.cod_modelo LEFT OUTER JOIN r_endosso2_:xxx ON r_apolice_:xxx.cod_apo = r_endosso2_:xxx.cod_apo AND r_apolice_:xxx.cod_modelo = r_endosso2_:xxx.cod_modelo;
'''

sql={}
for tab in tables:
    sql['tab'] = sql_code
    sql['tab'] = sql['tab'].replace(':xxx', tab)
    cur.execute(sql['tab'])
    print('Tabela rs_data_' + tab + ' carregada com sucesso')

conn.commit()
cur.close()
conn.close()
