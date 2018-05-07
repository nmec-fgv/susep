################################################################################
## Python database creation form data transformed pickled form data_transf.py ##
################################################################################


import os
import pickle
import numpy as np


def file_load(filename):
    try:
        os.path.exists('/home/pgsqldata/Susep/' + filename)
        with open('/home/pgsqldata/Susep/' + filename, 'rb') as file:
            x = pickle.load(file)
    except:
        print('File ' + filename + ' not found')

    return x


class Data:
    '''
    Data preparation for subsequent modeling.
    Loads data from files 'Data/data_mmmaa_transf.pkl' according to period request.
    Returns attribute .data, a dictionary containing 'X_exog', 'y_cas', 'y_rcd', 'y_app' and 'y_out', where y_* is divided in 'y_count' and 'y_claim'.
    
    Parameters:
    ----------
    period, takes 'mmm + aa' string value or '#tr + aa'
    threshold, dict containing keys 'cas', 'rcd', 'app', 'out', which must be provided and set to zero if no threshold is intended.
    '''

    def __init__(self, period):
        
        periods = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez', '1tr', '2tr', '3tr', '4tr')
        years = ('08', '09', '10', '11')

        if period[:3] not in periods and period[3:] not in years:
            raise Exception('period invalid or outside permissible range')

        if period[:3] in periods[:12]:
            (mmm, aa) = (period[:3], period[3:])
            filename = 'data_' + mmm + aa + '_transf.pkl'
            data = file_load(filename)
            data['X'] = data['X'].tolist()

        elif period[:3] in periods[12:]:
            aux = {}
            aa = period[3:]
            if period[0] == '1':
                for i, mmm in enumerate(periods[:3]):
                    filename = 'data_' + mmm + aa + '_transf.pkl'
                    aux[str(i)] = file_load(filename)
            elif period[0] == '2':
                for i, mmm in enumerate(periods[3:6]):
                    filename = 'data_' + mmm + aa + '_transf.pkl'
                    aux[str(i)] = file_load(filename)
            elif period[0] == '3':
                for i, mmm in enumerate(periods[6:9]):
                    filename = 'data_' + mmm + aa + '_transf.pkl'
                    aux[str(i)] = file_load(filename)
            elif period[0] == '4':
                for i, mmm in enumerate(periods[9:12]):
                    filename = 'data_' + mmm + aa + '_transf.pkl'
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

        def convert_arr(X, y):
            '''
            Internal auxiliary function.
            Takes X, y list argument and computes arrays for counts and claims.
            '''

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
        
        aux = {}
        aux['cas'] = convert_arr(data['X'], data['y_cas'])
        aux['rcd'] = convert_arr(data['X'], data['y_rcd'])
        aux['app'] = convert_arr(data['X'], data['y_app'])
        aux['out'] = convert_arr(data['X'], data['y_out'])

        self.period = period
        self.data = dict([('cas', aux['cas']), ('rcd', aux['rcd']), ('app', aux['app']), ('out', aux['out'])])
    
    def makefile(self):
        '''Saves data to pickle file'''

        mmm = self.period[:3]
        aa = self.period[3:]
        try:
            os.remove('/home/pgsqldata/Susep/data_' + mmm + aa + '_final.pkl')
        except OSError:
            pass
    
        with open('/home/pgsqldata/Susep/data_' + mmm + aa + '_final.pkl', 'wb') as file:
            pickle.dump(self.data, file)

        print('File data_' + mmm + aa + '_final.pkl saved') 


if __name__ == '__main__':
#    periods = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez', '1tr', '2tr', '3tr', '4tr')
#    periods = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')
    periods = ('fev',)
#    years = ('08', '09', '10', '11')
    years = ('08',)
    for period in periods:
        for aa in years:
            x = Data(period+aa)
            x.makefile()
