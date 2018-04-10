##############################################################
## Reparticionamento das tabelas rs_data em tabelas mensais ##
##############################################################

import psycopg2

conn = psycopg2.connect("dbname=susep user=ricardob")
cur = conn.cursor()


def gen_rs_tables(year):
    """Generates dictionary containing code for creation of rs_mmmaa tables."""

    months = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')
    sql_code = r'''
    DROP TABLE IF EXISTS rs_:mmmaa;

    CREATE TABLE rs_:mmmaa (
        cod_apo         varchar(20),
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
        sinistro        json,
        endosso         json
    );
    '''

    sql = {}
    for mmm in months:
        sql[mmm+year] = sql_code.replace(':mmmaa', mmm + year)
    return sql


def rs_tables(year, sem=False):
    """Generates dictionary containing codes that fill values in rs_mmmaa tables from rs_data_xxx tables."""

    if sem == False:
        months = [('jan', '01'), ('fev', '02'), ('mar', '03'), ('abr', '04'), ('mai', '05'), ('jun', '06'), ('jul', '07'), ('ago', '08'), ('set', '09'), ('out', '10'), ('nov', '11'), ('dez', '12')]
    else:
        months = [('jan', '01'), ('fev', '02'), ('mar', '03'), ('abr', '04'), ('mai', '05'), ('jun', '06')]

    if year == '16':
        if sem == False:
            tables = ('16b',)
        else:
            tables = ('16a',)
    elif year == '15':
        if sem == False:
            tables = ('15b', '16a', '16b')
        else:
            tables = ('15a',)
    elif year == '14':
        if sem == False:
            tables = ('14b', '15a', '15b', '16a', '16b')
        else:
            tables = ('14a',)
    elif year == '13':
        if sem == False:
            tables = ('13b', '14a', '14b', '15a', '15b', '16a', '16b')
        else:
            tables = ('14a',)
    elif year == '12':
        if sem == False:
            tables = ('12b', '13a', '13b', '14a', '14b', '15a', '15b', '16a', '16b')
        else:
            tables = ('12a',)
    elif year == '11':
        if sem == False:
            tables = ('11b', '12a', '12b', '13a', '13b', '14a', '14b', '15a', '15b', '16a', '16b')
        else:
            tables = ('11a',)
    elif year == '10':
        if sem == False:
            tables = ('10b', '11a', '11b', '12a', '12b', '13a', '13b', '14a', '14b', '15a', '15b', '16a')
        else:
            tables = ('10a',)
    elif year == '09':
        if sem == False:
            tables = ('09b', '10a', '10b', '11a', '11b', '12a', '12b', '13a', '13b', '14a', '14b', '15a')
        else:
            tables = ('09a',)
    elif year == '08':
        if sem == False:
            tables = ('08b', '09a', '09b', '10a', '10b', '11a', '11b', '12a', '12b', '13a', '13b', '14a')
    elif year == '07':
        if sem == False:
            tables = ('08b', '09a', '09b', '10a', '10b', '11a', '11b', '12a', '12b', '13a')
    elif year == '06':
        if sem == False:
            tables = ('08b', '09a', '09b', '10a', '10b', '11a', '11b', '12a')

    sql_code = r'''
    INSERT INTO rs_:mmmaa (
        cod_apo,
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
        sinistro,
        endosso
    )
    SELECT
        cod_apo,
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
        sinistro,
        endosso
    FROM rs_data_:xxx
    WHERE date_part('month', inicio_vig) = :mm AND date_part('year', inicio_vig) = 20:aa;
    '''

    sql2 = {}
    for mmm in months:
        for tab in tables:
            sql2[mmm[0]+year+tab] = sql_code.replace(':mmmaa', mmm[0] + year)
            sql2[mmm[0]+year+tab] = sql2[mmm[0]+year+tab].replace(':xxx', tab)
            sql2[mmm[0]+year+tab] = sql2[mmm[0]+year+tab].replace(':mm', mmm[1])
            sql2[mmm[0]+year+tab] = sql2[mmm[0]+year+tab].replace(':aa', year)
    return sql2


years = ('06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16')
years2 = years[3:]

for year in years:
    sql = gen_rs_tables(year)
    for i in sql:
        cur.execute(sql[i])
        print('Tabela rs_' + i + ' criada com sucesso')
    sql2 = rs_tables(year, sem=False)
    for i in sql2:
        cur.execute(sql2[i])
        print('Tabela rs_' + i[0:5] + ' carregada com dados da tabela rs_data_' + i[5:])

for year in years2:
    sql3 = rs_tables(year, sem=True)
    for i in sql3:
        cur.execute(sql3[i])
        print('Tabela rs_' + i[0:5] + ' carregada com dados da tabela rs_data_' + i[5:])


conn.commit()
cur.close()
conn.close()
