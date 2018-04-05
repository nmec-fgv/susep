/*-----------------------------------------------
Código para importação de arquivos S_AUTO_xxx.csv
-----------------------------------------------*/

\timing

/* Inserir período. Nota importante: deve-se modificar manualmente o período no
bloco de cópia de arquivos csv abaixo */
\set xxx 16b

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
    cep             integer /* following is 16A specific
    cod_seg_apolice varchar(30),
    id_calculado    varchar(10) */
);

/* Following code block requires manual updating of time period */
\copy s_auto_16b from 'C:\\Users\\Angel\\Susep\\Data\\16B\\S_AUTO_2016B.csv' delimiter ';' csv header null 'NULL';
