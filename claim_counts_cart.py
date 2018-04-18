#############################################
## Count data extraction from pickle files ##
#############################################

import os
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta


def load_pkl(filename):
    try:
        os.path.exists('Data/' + filename)
        with open('Data/' + filename, 'rb') as file:
            data = pickle.load(file)
    except:
        print('File ' + filename + ' not found')
    return data

def count_exposure(data):
    '''Returns count of sinisters, count of others and total exposure for sinisters for a given month/year'''

    max_count = 0
    for x in data:
        if x[2] != None:
            if len(x[2]) > max_count:
                max_count = len(x[2])

    count = {}
    count_outros = {}
    tot_exp = {}
    k =[]
    d = []
    for i in range(max_count+1):
        count[str(i)] = 0
        count_outros[str(i)] = 0
        tot_exp[str(i)] = 0

    for x in data:
        if x[2] == None and x[3] == None:
            count['0'] += 1
            count_outros['0'] += 1
            k.append(0)
            delta_years = 0
            delta_years += relativedelta(x[1], x[0]).years
            delta_years += relativedelta(x[1], x[0]).months / 12
            delta_years += relativedelta(x[1], x[0]).days / 365.2425
            tot_exp['0'] += delta_years
            d.append(delta_years)
        elif x[2] == None and x[3] != None:
            count['0'] += 1
            count_outros['0'] += 1
            fim_vig = x[1]
            for i in x[3].values():
                if i['f1'] in {'2', '3'}:
                    if (datetime.strptime(i['f2'], '%Y-%m-%d').date()-x[0]).days < 0:
                        fim_vig = x[0]
                    else:
                        fim_vig = datetime.strptime(i['f2'], '%Y-%m-%d').date()

            k.append(0)
            delta_years = 0
            delta_years += relativedelta(fim_vig, x[0]).years
            delta_years += relativedelta(fim_vig, x[0]).months / 12
            delta_years += relativedelta(fim_vig, x[0]).days / 365.2425
            tot_exp['0'] += delta_years
            d.append(delta_years)
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
            k.append(sin_count)
            delta_years = 0
            delta_years += relativedelta(x[1], x[0]).years
            delta_years += relativedelta(x[1], x[0]).months / 12
            delta_years += relativedelta(x[1], x[0]).days / 365.2425
            tot_exp[str(sin_count)] += delta_years
            d.append(delta_years)
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
                    if (datetime.strptime(i['f2'], '%Y-%m-%d').date()-x[0]).days < 0:
                        fim_vig = x[0]
                    else:
                        fim_vig = datetime.strptime(i['f2'], '%Y-%m-%d').date()
            
            count[str(sin_count)] += 1
            count_outros[str(sin_count_outros)] += 1
            k.append(sin_count)
            delta_years = 0
            delta_years += relativedelta(fim_vig, x[0]).years
            delta_years += relativedelta(fim_vig, x[0]).months / 12
            delta_years += relativedelta(fim_vig, x[0]).days / 365.2425
            tot_exp[str(sin_count)] += delta_years
            d.append(delta_years)

    return (count, count_outros, tot_exp, k, d)


if __name__ == "__main__":
    months = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')
    years = ('06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16')
    for mmm in months:
        for aa in years:
            filename = 'freq_dat_' + mmm + aa + '_cart.pkl'
            data = load_pkl(filename)
            results = count_exposure(data)

            try:
                os.remove('Data/cc_cart_' + mmm + aa + '.pkl')
            except OSError:
                pass
        
            with open('Data/cc_cart_' + mmm + aa + '.pkl', 'wb') as file:
                pickle.dump(results, file)

            print('File cc_cart_' + mmm + aa + '.pkl saved') 
