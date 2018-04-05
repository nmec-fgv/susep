import psycopg2
import os
import pickle

conn = psycopg2.connect("dbname='susep' user='ricardob' host='localhost' password = 'Rafa1201$' port='5432'")
cur = conn.cursor()

cur.execute('''SELECT * FROM s_auto2_16b;''')
data_sinistro = cur.fetchall()

conn.commit()
cur.close()
conn.close()

try:
    os.remove('data_sinistro.pkl')
except OSError:
    pass

with open('data_sinistro.pkl', 'wb') as file:
    pickle.dump(data_sinistro, file)
