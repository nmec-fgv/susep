import psycopg2
import os
import pickle

conn = psycopg2.connect("dbname=susep user=ricardob")
cur = conn.cursor()

cur.execute('''SELECT * FROM rs_data_16b;''')
rs_data = cur.fetchall()

conn.commit()
cur.close()
conn.close()

try:
    os.remove('rs_data.pkl')
except OSError:
    pass

with open('rs_data.pkl', 'wb') as file:
    pickle.dump(rs_data, file)
