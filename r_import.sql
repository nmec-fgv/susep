/*-----------------------------------------------------------------------------
Código para importação de arquivos R_AUTO_xxx.csv e geração de tabelas r_auto2.
-----------------------------------------------------------------------------*/

\timing

/* Inserir período. Nota importante: deve-se modificar manualmente o período no
bloco de cópia de arquivos csv abaixo */
\set xxx 16b

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
    sinal           varchar(10)  /*Columns below are specific to 16A
    cod_seg_apolice varchar(30),
    id_calculado    varchar(10)  */
);

/* Following code block requires manual updating of time period */
\copy r_auto_16b from 'C:\\Users\\Angel\\Susep\\Data\\16B\\R_AUTO_2016B-000.csv' delimiter ';' csv header null 'NULL';
\copy r_auto_16b from 'C:\\Users\\Angel\\Susep\\Data\\16B\\R_AUTO_2016B-001.csv' delimiter ';' csv header null 'NULL';
\copy r_auto_16b from 'C:\\Users\\Angel\\Susep\\Data\\16B\\R_AUTO_2016B-002.csv' delimiter ';' csv header null 'NULL';
\copy r_auto_16b from 'C:\\Users\\Angel\\Susep\\Data\\16B\\R_AUTO_2016B-003.csv' delimiter ';' csv header null 'NULL';
\copy r_auto_16b from 'C:\\Users\\Angel\\Susep\\Data\\16B\\R_AUTO_2016B-004.csv' delimiter ';' csv header null 'NULL';
\copy r_auto_16b from 'C:\\Users\\Angel\\Susep\\Data\\16B\\R_AUTO_2016B-005.csv' delimiter ';' csv header null 'NULL';
\copy r_auto_16b from 'C:\\Users\\Angel\\Susep\\Data\\16B\\R_AUTO_2016B-006.csv' delimiter ';' csv header null 'NULL';
\copy r_auto_16b from 'C:\\Users\\Angel\\Susep\\Data\\16B\\R_AUTO_2016B-007.csv' delimiter ';' csv header null 'NULL';
\copy r_auto_16b from 'C:\\Users\\Angel\\Susep\\Data\\16B\\R_AUTO_2016B-008.csv' delimiter ';' csv header null 'NULL';
\copy r_auto_16b from 'C:\\Users\\Angel\\Susep\\Data\\16B\\R_AUTO_2016B-009.csv' delimiter ';' csv header null 'NULL';
\copy r_auto_16b from 'C:\\Users\\Angel\\Susep\\Data\\16B\\R_AUTO_2016B-010.csv' delimiter ';' csv header null 'NULL';
--\copy r_auto_12a from 'C:\\Users\\Angel\\Susep\\Data\\12A\\R_AUTO_2012A-011.csv' delimiter ';' csv header null 'NULL';
--\copy r_auto_12b from 'C:\\Users\\Angel\\Susep\\Data\\12B\\R_AUTO_2012B-012.csv' delimiter ';' csv header null 'NULL';
