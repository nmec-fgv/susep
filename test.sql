/* Postgres testing code */

DROP TABLE IF EXISTS test_r;
DROP TABLE IF EXISTS test_s;
DROP TABLE IF EXISTS autodata_test;

CREATE TABLE test_r (cod_apo varchar(20), item varchar(20));
INSERT INTO test_r (cod_apo, item) VALUES ('001', 'first'), ('002', 'second');
CREATE TABLE test_s (cod_apo varchar(20), evento varchar(20));
INSERT INTO test_s (cod_apo, evento) VALUES ('002', 'sinistro1'), ('002', 'sinistro2');

CREATE TABLE autodata_test (cod_apo varchar(20), item varchar(20), evento1 varchar(20) DEFAULT NULL, evento2 varchar(20) DEFAULT NULL);
INSERT INTO autodata_test (cod_apo, item) SELECT cod_apo, item FROM test_r;
--INSERT INTO autodata_test (
