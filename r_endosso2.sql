/* Código para transformação das tabelas r_endosso */

\timing
\set xxx 16b

DROP TABLE IF EXISTS r_endosso2_:xxx;

CREATE TABLE r_endosso2_:xxx
    AS SELECT
        cod_apo,
        cod_modelo,
        json_object_agg(endosso, (cod_end, inicio_vig)) AS endosso
        FROM r_endosso_:xxx
        GROUP BY cod_apo, cod_modelo;
