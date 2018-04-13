#############################################
## Count data extraction from pickle files ##
#############################################

import os
import pickle
from datetime import datetime


def load_pkl(filename):
    try:
        os.path.exists('/home/ricardob/Susep/Data/' + filename)
        with open('/home/ricardob/Susep/Data/' + filename, 'rb') as file:
            data = pickle.load(file)
    except:
        print('File ' + filename + ' not found')
    return data


def count_exposure(data):
    '''Returns count of sinisters, count of others and total exposure for sinisters for a given month/year.'''
    max_count = 0
    for x in data:
        if x[2] != None:
            if len(x[2]) > max_count:
                max_count = len(x[2])

    count = {}
    count_outros = {}
    tot_exp = {}
    for i in range(max_count+1):
        count[str(i)] = 0
        count_outros[str(i)] = 0
        tot_exp[str(i)] = 0

    for x in data:
        if x[2] == None and x[3] == None:
            count['0'] += 1
            count_outros['0'] += 1
            tot_exp['0'] += (x[1] - x[0]).days
        elif x[2] == None and x[3] != None:
            count['0'] += 1
            count_outros['0'] += 1
            fim_vig = x[1]
            for i in x[3].values():
                if i['f1'] in {'2', '3'}:
                    fim_vig = datetime.strptime(i['f2'], '%Y-%m-%d').date()
            tot_exp['0'] += (fim_vig - x[0]).days
        elif x[2] != None and x[3] == None:
            sin_count = 0
            sin_count_outros = 0
            for i in x[2]:
                if i in {'1', '2', '3', '4', '5', '6'}:
                    sin_count += 1
                elif i in {7, 8, 9}:
                    sin_count_outros += 1
            count[str(sin_count)] += 1
            count_outros[str(sin_count_outros)] += 1
            tot_exp[str(sin_count)] += (x[1] - x[0]).days
        else:
            sin_count = 0
            sin_count_outros = 0
            fim_vig = x[1]
            for i in x[2]:
                if i in {'1', '2', '3', '4', '5', '6'}:
                    sin_count += 1
                elif i in {7, 8, 9}:
                    sin_count_outros += 1
            for i in x[3].values():
                if i['f1'] in {'2', '3'}:
                    fim_vig = datetime.strptime(i['f2'], '%Y-%m-%d').date()
            count[str(sin_count)] += 1
            count_outros[str(sin_count_outros)] += 1
            tot_exp[str(sin_count)] += (fim_vig - x[0]).days

    for i in range(max_count+1):
        tot_exp[str(i)] = tot_exp[str(i)] / 365.2425

    return (count, count_outros, tot_exp)


if __name__ == "__main__":
    months = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')
    years = ('06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16')
    results = {}
    for mmm in months:
        for aa in years:
            filename = 'freq_dat_' + mmm + aa + '.pkl'
            data = load_pkl(filename)
            results[mmm+aa] = count_exposure(data)

        try:
            os.remove('/home/ricardob/Susep/Data/claim_counts.pkl')
        except OSError:
            pass

        with open('/home/ricardob/Susep/Data/claim_counts.pkl', 'wb') as file:
            pickle.dump(results, file)
