################################################################################
## Python database creation form data transformed pickled form data_transf.py ##################################################################################


import os
import pickle
import numpy as np
import shelve
import pdb


def file_load(filename):
    try:
        os.path.exists('Data/' + filename)
        with open('Data/' + filename, 'rb') as file:
            res = pickle.load(file)
    except:
        print('File ' + filename + ' not found')

    return res


class Data:
    '''
    Data preparation for subsequent modeling.
    Loads data from files 'Data/data_mmmaa.pkl' according to period request.
    Returns attribute .data, a dictionary containing 'X_exog', 'y_cas', 'y_rcd', 'y_app' and 'y_out', where y_* is divided in 'y_count' and 'y_claim'.
    
    Parameters:
    ----------
    period, takes 'mmm + aa' string value or '#tr + aa'
    threshold, dict containing keys 'cas', 'rcd', 'app', 'out', which must be provided and set to zero if no threshold is intended.
    '''

    def __init__(self, period, threshold):
        
        periods = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez', '1tr', '2tr', '3tr', '4tr')
        years = ('08', '09', '10', '11')

        if period[:3] not in periods and period[3:] not in years:
            raise Exception('period invalid or outside permissible range')

        for item in threshold.values():
            if isinstance(item, (int, float)) == False:
                raise Exception('threshold invalid, provide permissible dictionary object')

        if period[:3] in periods[:12]:
            (mmm, aa) = (period[:3], period[3:])
            filename = 'data_' + mmm + aa + '.pkl'
            data = file_load(filename)
            data['X'] = data['X'].tolist()

        elif period[:3] in periods[12:]:
            aux = {}
            aa = period[3:]
            if period[0] == '1':
                for i, mmm in enumerate(periods[:3]):
                    filename = 'data_' + mmm + aa + '.pkl'
                    aux[str(i)] = file_load(filename)
            elif period[0] == '2':
                for i, mmm in enumerate(periods[3:6]):
                    filename = 'data_' + mmm + aa + '.pkl'
                    aux[str(i)] = file_load(filename)
            elif period[0] == '3':
                for i, mmm in enumerate(periods[6:9]):
                    filename = 'data_' + mmm + aa + '.pkl'
                    aux[str(i)] = file_load(filename)
            elif period[0] == '4':
                for i, mmm in enumerate(periods[9:12]):
                    filename = 'data_' + mmm + aa + '.pkl'
                    aux[str(i)] = file_load(filename)

            data = {}
            data['y_cas'] = []
            data['y_rcd'] = []
            data['y_app'] = []
            data['y_out'] = []
            data['X'] = []
            for i in range(3):
                data['y_cas'] += aux[str(i)]['y_cas']
                data['y_rcd'] += aux[str(i)]['y_rcd']
                data['y_app'] += aux[str(i)]['y_app']
                data['y_out'] += aux[str(i)]['y_out']
                for item in aux[str(i)]['X']:
                    data['X'].append(item)

        def convert_arr(X, y, y_threshold):
            '''
            Internal auxiliary function.
            Takes X, y list argument and computes arrays for counts and claims.
            '''

            if y_threshold == 0:
                y_count = []
                y_claim = []
                X_claim = []
                for i, item in enumerate(y):
                    y_count.append(len(item))
                    if len(item) > 0:
                        for j in range(len(item)):
                            y_claim.append(item[j])
                            X_claim.append(X[i])
                    else:
                        y_claim.append(0)
                        X_claim.append(X[i])

                return dict([('X_count', X), ('X_claim', X_claim), ('y_count', y_count), ('y_claim', y_claim)])
        
            else:
                y_count_mc = []
                y_count_ec = []
                y_claim_mc = []
                y_claim_ec = []
                X_claim_mc = []
                X_claim_ec = []
                for i, item in enumerate(y):
                    y_count_mc.append(sum(x < y_threshold for x in item))
                    y_count_ec.append(sum(x >= y_threshold for x in item))
                    if len([x for x in item if x < y_threshold]) > 0:
                        for j in range(len([x for x in item if x < y_threshold])):
                            y_claim_mc.append([x for x in item if x < y_threshold][j])
                            X_claim_mc.append(X[i])
                    else:
                        y_claim_mc.append(0)
                        X_claim_mc.append(X[i])
                    
                    if len([x for x in item if x >= y_threshold]) > 0:
                        for j in range(len([x for x in item if x >= y_threshold])):
                            y_claim_ec.append([x for x in item if x >= y_threshold][j])
                            X_claim_ec.append(X[i])
                    else:
                        y_claim_ec.append(0)
                        X_claim_ec.append(X[i])
                    
                return dict([('X_count_mc', X), ('X_count_ec', X), ('X_claim_mc', X_claim_mc), ('X_claim_ec', X_claim_ec), ('y_count_mc', y_count_mc), ('y_count_ec', y_count_ec), ('y_claim_mc', y_claim_mc), ('y_claim_ec', y_claim_ec)])
    
        aux = {}
        aux['cas'] = convert_arr(data['X'], data['y_cas'], threshold['cas'])
        aux['rcd'] = convert_arr(data['X'], data['y_rcd'], threshold['rcd'])
        aux['app'] = convert_arr(data['X'], data['y_app'], threshold['app'])
        aux['out'] = convert_arr(data['X'], data['y_out'], threshold['out'])

        self.data = dict([('cas', aux['cas']), ('rcd', aux['rcd']), ('app', aux['app']), ('out', aux['out'])])


if __name__ == '__main__':
#    periods = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez', '1tr', '2tr', '3tr', '4tr')
#    periods = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')
    periods = ('1tr',)
#    years = ('08', '09', '10', '11')
    years = ('09',)
    threshold = dict([('cas', 0), ('rcd', 0), ('app', 0), ('out', 0)])
    db_aux = {}
    for period in periods:
        for aa in years:
            db_aux[period+aa] = Data(period+aa, threshold)
            print('Data for period ' + period + aa + ' stored in dict')

#    db_file = 'db_thres0_qtrly09'
#    db = shelve.open('Data/' + db_file)
#    for key, item in zip(db_aux.keys(), db_aux.values()):
#        db[key] = item
#    db.close()
