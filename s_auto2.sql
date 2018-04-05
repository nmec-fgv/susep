/* Código para transformação das tabelas s_auto */

\timing
\set xxx 16b

DROP TABLE IF EXISTS s_auto2_:xxx;

CREATE TABLE s_auto2_:xxx
    AS SELECT
        cod_apo,
        cod_modelo,
        json_object_agg(evento, (indeniz, d_ocorr)) AS sinistro
        FROM s_auto_:xxx
        GROUP BY cod_apo, cod_modelo;
