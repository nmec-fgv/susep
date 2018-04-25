###########################################################
## Código para inserção de valores nas tabelas r_apolice ##
###########################################################

import os
import psycopg2

conn = psycopg2.connect("dbname=susep user=ricardob")
cur = conn.cursor()

tables = ('08B', '09A', '09B', '10A', '10B', '11A', '11B', '12A', '12B', '13A', '13B', '14A', '14B', '15A', '15B', '16A', '16B')

sql_code = r'''
DROP TABLE IF EXISTS r_apolice_:xxx;

CREATE TABLE r_apolice_:xxx (
    cod_apo,
    endosso,
    cod_end,
    item,
    tipo_pes,
    modalidade,
    tipo_prod,
    cobertura,
    cod_modelo,
    ano_modelo,
    cod_tarif,
    regiao,
    cod_cont,
    tipo_franq,
    val_franq,
    perc_fator,
    tab_ref,
    is_casco,
    is_rcdmat,
    is_rcdc,
    is_rcdmor,
    is_app_ma,
    is_app_ipa,
    is_app_dmh,
    pre_casco,
    pre_casco_co,
    pre_rcdmat,
    pre_rcdc,
    pre_rcdmor,
    pre_app_ma,
    pre_app_ia,
    pre_app_dm,
    pre_outros,
    inicio_vig,
    fim_vig,
    perc_bonus,
    clas_bonus,
    perc_corr,
    sexo,
    data_nasc,
    tempo_hab,
    utilizacao,
    cep_util,
    cep_per
)
AS SELECT
    cod_apo,
    endosso,
    cod_end,
    item,
    tipo_pes,
    modalidade,
    tipo_prod,
    cobertura,
    cod_modelo,
    ano_modelo,
    cod_tarif,
    regiao,
    cod_cont,
    tipo_franq,
    val_franq,
    perc_fator,
    tab_ref,
    is_casco,
    is_rcdmat,
    is_rcdc,
    is_rcdmor,
    is_app_ma,
    is_app_ipa,
    is_app_dmh,
    pre_casco,
    pre_casco_co,
    pre_rcdmat,
    pre_rcdc,
    pre_rcdmor,
    pre_app_ma,
    pre_app_ia,
    pre_app_dm,
    pre_outros,
    inicio_vig,
    fim_vig,
    perc_bonus,
    clas_bonus,
    perc_corr,
    sexo,
    data_nasc,
    tempo_hab,
    utilizacao,
    cep_util,
    cep_per
FROM r_auto_:xxx
WHERE endosso = '0000000000';
'''

sql_code2 = r'''
DROP TABLE IF EXISTS r_apolice_:xxx;

CREATE TABLE r_apolice_:xxx (
    cod_apo,
    endosso,
    cod_end,
    item,
    tipo_pes,
    modalidade,
    tipo_prod,
    cobertura,
    cod_modelo,
    ano_modelo,
    cod_tarif,
    regiao,
    cod_cont,
    tipo_franq,
    val_franq,
    perc_fator,
    tab_ref,
    is_casco,
    is_rcdmat,
    is_rcdc,
    is_rcdmor,
    is_app_ma,
    is_app_ipa,
    is_app_dmh,
    pre_casco,
    pre_casco_co,
    pre_rcdmat,
    pre_rcdc,
    pre_rcdmor,
    pre_app_ma,
    pre_app_ia,
    pre_app_dm,
    pre_outros,
    inicio_vig,
    fim_vig,
    perc_bonus,
    clas_bonus,
    perc_corr,
    sexo,
    data_nasc,
    tempo_hab,
    utilizacao,
    cep_util,
    cep_per,
    cod_seg_apolice,
    id_calculado
)
AS SELECT
    cod_apo,
    endosso,
    cod_end,
    item,
    tipo_pes,
    modalidade,
    tipo_prod,
    cobertura,
    cod_modelo,
    ano_modelo,
    cod_tarif,
    regiao,
    cod_cont,
    tipo_franq,
    val_franq,
    perc_fator,
    tab_ref,
    is_casco,
    is_rcdmat,
    is_rcdc,
    is_rcdmor,
    is_app_ma,
    is_app_ipa,
    is_app_dmh,
    pre_casco,
    pre_casco_co,
    pre_rcdmat,
    pre_rcdc,
    pre_rcdmor,
    pre_app_ma,
    pre_app_ia,
    pre_app_dm,
    pre_outros,
    inicio_vig,
    fim_vig,
    perc_bonus,
    clas_bonus,
    perc_corr,
    sexo,
    data_nasc,
    tempo_hab,
    utilizacao,
    cep_util,
    cep_per,
    cod_seg_apolice,
    id_calculado
FROM r_auto_:xxx
WHERE endosso = '0000000000';
'''

sql={}
for tab in tables:
    if tab == '16A':
        sql['tab'] = sql_code2
    else:
        sql['tab'] = sql_code
    sql['tab'] = sql['tab'].replace(':xxx', tab)
    cur.execute(sql['tab'])
    print('Tabela r_apolice_' + tab + ' carregada com sucesso')

conn.commit()
cur.close()
conn.close()
