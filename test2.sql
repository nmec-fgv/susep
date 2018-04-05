/* Example from https://stackoverflow.com/questions/15506199/dynamic-alternative-to-pivot-with-case-and-group-by/15514334#15514334 */

--DROP TABLE IF EXISTS test2;
--CREATE TABLE test2 (id integer, feh integer, bar varchar(2));
--INSERT INTO test2 (id, feh, bar) VALUES (1, 10, 'A'), (2, 20, 'A'), (3, 3, 'B'), (4, 4, 'B'), (5, 5, 'C'), (6, 6, 'D'), (7, 7, 'D'), (8, 8, 'D');

CREATE TEMP TABLE tbl (row_name text, attrib text, val int, val2 int);
INSERT INTO tbl (row_name, attrib, val, val2) VALUES
    ('A', 'val1', 10, 100),
    ('A', 'val2', 20, 23),
    ('B', 'val1', 3, 4),
    ('B', 'val2', 4, 5),
    ('C', 'val1', 5, 6),
    ('D', 'val3', 8, 80),
    ('D', 'val1', 6, 87),
    ('D', 'val2', 7, 9);

--SELECT bar,
--    MAX(CASE WHEN abc."row" = 1 THEN feh ELSE NULL END) AS "val1",
--    MAX(CASE WHEN abc."row" = 2 THEN feh ELSE NULL END) AS "val2",
--    MAX(CASE WHEN abc."row" = 3 THEN feh ELSE NULL END) AS "val3"
--FROM
--(
--    SELECT bar, feh, row_number() OVER (partition by bar) as row
--    FROM test2
-- ) abc
--GROUP BY bar;

--SELECT * FROM crosstab(
--    'SELECT    bar, 1 AS cat, feh
--     FROM      test2
--     ORDER BY  bar, feh')
-- AS ct (bar varchar(2), val1 integer, val2 integer, val3 integer);

DROP TABLE IF EXISTS test2;
CREATE TABLE test2
AS SELECT
    row_name AS bar,
    json_object_agg(attrib, (val, val2)) AS data
FROM tbl
GROUP BY row_name;
--ORDER BY row_name;
