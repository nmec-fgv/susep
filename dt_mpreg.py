#######################################################
## Data transformation for mixed poisson regressions ##
#######################################################

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

def data_transf(data):
    '''
    Transformation of database records for regression purposes
    
    Data encoding:
    res[0] -> dummy dif(ano modelo, ano apolice) = 1
    res[1] -> dummy dif(ano modelo, ano apolice) = 2
    res[2] -> dummy dif(ano modelo, ano apolice) = 3
    res[3] -> dummy dif(ano modelo, ano apolice) = 4
    res[4] -> dummy dif(ano modelo, ano apolice) = 5
    res[5] -> dummy dif(ano modelo, ano apolice) in [6,10]
    res[6] -> dummy dif(ano modelo, ano apolice) in [11,20]
    res[7] -> dummy dif(ano modelo, ano apolice) > 20

    res[k] -> k variable - claims count
    res[d] -> d variable - exposure in years
    '''

    res = [0] * d # change d to list range
    for x in data:
        if x[1] >= 2000 + int(aa) 
            pass
        elif (2000 + int(aa) - x[1]) == 1:
            res[0] = 1
        elif (2000 + int(aa) - x[1]) == 2:
            res[1] = 1
        elif (2000 + int(aa) - x[1]) == 3:
            res[2] = 1
        elif (2000 + int(aa) - x[1]) == 4:
            res[3] = 1
        elif (2000 + int(aa) - x[1]) == 5:
            res[4] = 1
        elif (2000 + int(aa) - x[1]) >= 6 and (2000 + int(aa) - x[1]) <= 10:
            res[5] = 1
        elif (2000 + int(aa) - x[1]) >= 11 and (2000 + int(aa) - x[1]) <= 20:
            res[6] = 1
        elif (2000 + int(aa) - x[1]) > 20:
            res[7] = 1



        if x[8] == None and x[9] == None:
            k.append(0)
            delta_years = 0
            delta_years += relativedelta(x[5], x[4]).years
            delta_years += relativedelta(x[5], x[4]).months / 12
            delta_years += relativedelta(x[5], x[4]).days / 365.2425
            d.append(delta_years)
        elif x[8] == None and x[9] != None:
            fim_vig = x[5]
            for i in x[9].values():
                if i['f1'] in {'2', '3'}:
                    if (datetime.strptime(i['f2'], '%Y-%m-%d').date()-x[4]).days < 0:
                        fim_vig = x[4]
                    else:
                        fim_vig = datetime.strptime(i['f2'], '%Y-%m-%d').date()

            k.append(0)
            delta_years = 0
            delta_years += relativedelta(fim_vig, x[4]).years
            delta_years += relativedelta(fim_vig, x[4]).months / 12
            delta_years += relativedelta(fim_vig, x[4]).days / 365.2425
            d.append(delta_years)
        elif x[8] != None and x[9] == None:
            sin_count = 0
            sin_count_outros = 0
            for i in x[8]:
                if i in {'1', '2', '3', '4', '5', '6'}:
                    sin_count += 1
                elif i in {7, 8, 9}:
                    sin_count_outros += 1

            k.append(sin_count)
            delta_years = 0
            delta_years += relativedelta(x[5], x[4]).years
            delta_years += relativedelta(x[5], x[4]).months / 12
            delta_years += relativedelta(x[5], x[4]).days / 365.2425
            d.append(delta_years)
        else:
            sin_count = 0
            sin_count_outros = 0
            fim_vig = x[5]
            for i in x[8]:
                if i in {'1', '2', '3', '4', '5', '6'}:
                    sin_count += 1
                elif i in {7, 8, 9}:
                    sin_count_outros += 1
            for i in x[9].values():
                if i['f1'] in {'2', '3'}:
                    if (datetime.strptime(i['f2'], '%Y-%m-%d').date()-x[4]).days < 0:
                        fim_vig = x[4]
                    else:
                        fim_vig = datetime.strptime(i['f2'], '%Y-%m-%d').date()
            
            k.append(sin_count)
            delta_years = 0
            delta_years += relativedelta(fim_vig, x[4]).years
            delta_years += relativedelta(fim_vig, x[4]).months / 12
            delta_years += relativedelta(fim_vig, x[4]).days / 365.2425
            d.append(delta_years)

    return (res)


if __name__ == "__main__":
    months = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')
    years = ('08', '09', '10', '11')
    for mmm in months:
        for aa in years:
            filename = 'data_mpreg_' + mmm + aa + '_raw.pkl'
            data = load_pkl(filename)
            results = data_transf(data)

            try:
                os.remove('Data/data_mpreg_' + mmm + aa + '.pkl')
            except OSError:
                pass
        
            with open('Data/data_mpreg_' + mmm + aa + '.pkl', 'wb') as file:
                pickle.dump(results, file)

            print('File data_mpreg_' + mmm + aa + '.pkl saved') 
