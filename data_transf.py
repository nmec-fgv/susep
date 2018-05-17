#########################################
## Data transformation for regressions ##
#########################################

import os
import pickle
import numpy as np
import re
import pdb
from datetime import datetime
from dateutil.relativedelta import relativedelta


def load_pkl(filename):
    try:
        os.path.exists('/home/pgsqldata/Susep/' + filename)
        with open('/home/pgsqldata/Susep/' + filename, 'rb') as file:
            data = pickle.load(file)
    except:
        print('File ' + filename + ' not found')
    return data

def data_transf(data):
    '''
    Transformation of database records for regression purposes
    
    Data encoding:
    base class: ano_modelo=0, passeio nac, regiao met sp, sexo masc, contrato V.M.R., classe bonus=0, tipo franquia=2 
    res[item][0] -> exposure in years
    res[item][1] -> constant 1
    res[item][2] -> dummy dif(ano modelo, ano apolice) = 1
    res[item][3] -> dummy dif(ano modelo, ano apolice) = 2
    res[item][4] -> dummy dif(ano modelo, ano apolice) = 3
    res[item][5] -> dummy dif(ano modelo, ano apolice) = 4
    res[item][6] -> dummy dif(ano modelo, ano apolice) = 5
    res[item][7] -> dummy dif(ano modelo, ano apolice) in [6,10]
    res[item][8] -> dummy dif(ano modelo, ano apolice) in [11,20]
    res[item][9] -> dummy dif(ano modelo, ano apolice) > 20
    res[item][10] -> dummy cod_tarif = 11
    res[item][11] -> dummy cod_tarif = 14A
    res[item][12] -> dummy cod_tarif = 14B
    res[item][13] -> dummy cod_tarif = 14C
    res[item][14] -> dummy cod_tarif = 15
    res[item][15] -> dummy cod_tarif = 16
    res[item][16] -> dummy cod_tarif = 17
    res[item][17] -> dummy cod_tarif = 18
    res[item][18] -> dummy cod_tarif = 19
    res[item][19] -> dummy cod_tarif = 20
    res[item][20] -> dummy cod_tarif = 21
    res[item][21] -> dummy cod_tarif = 22
    res[item][22] -> dummy cod_tarif = 23
    res[item][23] -> dummy regiao = 01
    res[item][24] -> dummy regiao = 02
    res[item][25] -> dummy regiao = 03
    res[item][26] -> dummy regiao = 04
    res[item][27] -> dummy regiao = 05
    res[item][28] -> dummy regiao = 06
    res[item][29] -> dummy regiao = 07
    res[item][30] -> dummy regiao = 08
    res[item][31] -> dummy regiao = 09
    res[item][32] -> dummy regiao = 10
    res[item][33] -> dummy regiao = 12
    res[item][34] -> dummy regiao = 13
    res[item][35] -> dummy regiao = 14
    res[item][36] -> dummy regiao = 15
    res[item][37] -> dummy regiao = 16
    res[item][38] -> dummy regiao = 17
    res[item][39] -> dummy regiao = 18
    res[item][40] -> dummy regiao = 19
    res[item][41] -> dummy regiao = 20
    res[item][42] -> dummy regiao = 21
    res[item][43] -> dummy regiao = 22
    res[item][44] -> dummy regiao = 23
    res[item][45] -> dummy regiao = 24
    res[item][46] -> dummy regiao = 25
    res[item][47] -> dummy regiao = 26
    res[item][48] -> dummy regiao = 27
    res[item][49] -> dummy regiao = 28
    res[item][50] -> dummy regiao = 29
    res[item][51] -> dummy regiao = 30
    res[item][52] -> dummy regiao = 31
    res[item][53] -> dummy regiao = 32
    res[item][54] -> dummy regiao = 33
    res[item][55] -> dummy regiao = 34
    res[item][56] -> dummy regiao = 35
    res[item][57] -> dummy regiao = 36
    res[item][58] -> dummy regiao = 37
    res[item][59] -> dummy regiao = 38
    res[item][60] -> dummy regiao = 39
    res[item][61] -> dummy regiao = 40
    res[item][62] -> dummy regiao = 41
    res[item][63] -> dummy sexo = F
    res[item][64] -> continuous idade
    res[item][65] -> dummy codigo_contrato = 2 (valor definido)
    res[item][66] -> dummy classe bonus = 1
    res[item][67] -> dummy classe bonus = 2
    res[item][68] -> dummy classe bonus = 3
    res[item][69] -> dummy classe bonus = 4
    res[item][70] -> dummy classe bonus = 5
    res[item][71] -> dummy classe bonus = 6
    res[item][72] -> dummy classe bonus = 7
    res[item][73] -> dummy classe bonus = 8
    res[item][74] -> dummy classe bonus = 9
    res[item][75] -> dummy tipo franquia = 1 (reduzida)
    res[item][76] -> dummy tipo franquia = 3 (majorada)
    res[item][77] -> dummy tipo franquia = 4 (dedutível)
    res[item][78] -> dummy tipo franquia = 9 (s/ franquia)
    res[item][79] -> continuous valor franquia
    res[item][80] -> continuous is_casco
    res[item][81] -> continuous is_rcdmat + is_rcdc + is_rcdmor
    res[item][82] -> continuous is_app_ma + is_app_ipa + is_app_dmh
    res[item][83] -> lista, valores indenizados casco
    res[item][84] -> lista, valores indenizados rcd
    res[item][85] -> lista, valores indenizados app
    res[item][86] -> lista, valores indenizados outros
    '''

    res = []
    for item, x in enumerate(data):
        res.append([0] * 87)
        res[item][1] = 1
        res[item][83] = []
        res[item][84] = []
        res[item][85] = []
        res[item][86] = []
        if x[1] >= 2000 + int(aa):
            pass
        elif (2000 + int(aa) - x[1]) == 1:
            res[item][2] = 1
        elif (2000 + int(aa) - x[1]) == 2:
            res[item][3] = 1
        elif (2000 + int(aa) - x[1]) == 3:
            res[item][4] = 1
        elif (2000 + int(aa) - x[1]) == 4:
            res[item][5] = 1
        elif (2000 + int(aa) - x[1]) == 5:
            res[item][6] = 1
        elif (2000 + int(aa) - x[1]) >= 6 and (2000 + int(aa) - x[1]) <= 10:
            res[item][7] = 1
        elif (2000 + int(aa) - x[1]) >= 11 and (2000 + int(aa) - x[1]) <= 20:
            res[item][8] = 1
        elif (2000 + int(aa) - x[1]) > 20:
            res[item][9] = 1

        if x[2] == ' 11' or x[2] == '11 ':
            res[item][10] = 1
        if x[2] == '14A':
            res[item][11] = 1
        if x[2] == '14B':
            res[item][12] = 1
        if x[2] == '14C':
            res[item][13] = 1
        if x[2] == ' 15' or x[2] == '15 ':
            res[item][14] = 1
        if x[2] == ' 16' or x[2] == '16 ':
            res[item][15] = 1
        if x[2] == ' 17' or x[2] == '17 ':
            res[item][16] = 1
        if x[2] == ' 18' or x[2] == '18 ':
            res[item][17] = 1
        if x[2] == ' 19' or x[2] == '19 ':
            res[item][18] = 1
        if x[2] == ' 20' or x[2] == '20 ':
            res[item][19] = 1
        if x[2] == ' 21' or x[2] == '21 ':
            res[item][20] = 1
        if x[2] == ' 22' or x[2] == '22 ':
            res[item][21] = 1
        if x[2] == ' 23' or x[2] == '23 ':
            res[item][22] = 1

        if x[3] == '01':
            res[item][23] = 1
        if x[3] == '02':
            res[item][24] = 1
        if x[3] == '03':
            res[item][25] = 1
        if x[3] == '04':
            res[item][26] = 1
        if x[3] == '05':
            res[item][27] = 1
        if x[3] == '06':
            res[item][28] = 1
        if x[3] == '07':
            res[item][29] = 1
        if x[3] == '08':
            res[item][30] = 1
        if x[3] == '09':
            res[item][31] = 1
        if x[3] == '10':
            res[item][32] = 1
        if x[3] == '12':
            res[item][33] = 1
        if x[3] == '13':
            res[item][34] = 1
        if x[3] == '14':
            res[item][35] = 1
        if x[3] == '15':
            res[item][36] = 1
        if x[3] == '16':
            res[item][37] = 1
        if x[3] == '17':
            res[item][38] = 1
        if x[3] == '18':
            res[item][39] = 1
        if x[3] == '19':
            res[item][40] = 1
        if x[3] == '20':
            res[item][41] = 1
        if x[3] == '21':
            res[item][42] = 1
        if x[3] == '22':
            res[item][43] = 1
        if x[3] == '23':
            res[item][44] = 1
        if x[3] == '24':
            res[item][45] = 1
        if x[3] == '25':
            res[item][46] = 1
        if x[3] == '26':
            res[item][47] = 1
        if x[3] == '27':
            res[item][48] = 1
        if x[3] == '28':
            res[item][49] = 1
        if x[3] == '29':
            res[item][50] = 1
        if x[3] == '30':
            res[item][51] = 1
        if x[3] == '31':
            res[item][52] = 1
        if x[3] == '32':
            res[item][53] = 1
        if x[3] == '33':
            res[item][54] = 1
        if x[3] == '34':
            res[item][55] = 1
        if x[3] == '35':
            res[item][56] = 1
        if x[3] == '36':
            res[item][57] = 1
        if x[3] == '37':
            res[item][58] = 1
        if x[3] == '38':
            res[item][59] = 1
        if x[3] == '39':
            res[item][60] = 1
        if x[3] == '40':
            res[item][61] = 1
        if x[3] == '41':
            res[item][62] = 1

        if x[6] == 'F':
            res[item][63] = 1

        res[item][64] += relativedelta(x[4], x[7]).years
        res[item][64] += relativedelta(x[4], x[7]).months / 12
        res[item][64] += relativedelta(x[4], x[7]).days / 365.2425

        if x[8] != None:
            aux8 = list(zip(re.findall(r' "(\d)" ', x[8]), re.findall(r'"f1":(\d+)', x[8]), re.findall(r'"f2":"(\d+-\d+-\d+)"', x[8])))
        if x[9] != None:
            aux9 = list(zip(re.findall(r'"f1":"(\d)"', x[9]), re.findall(r'"f2":"(\d+-\d+-\d+)"', x[9])))

        if x[8] == None and x[9] == None:
            delta_years = 0
            delta_years += relativedelta(x[5], x[4]).years
            delta_years += relativedelta(x[5], x[4]).months / 12
            delta_years += relativedelta(x[5], x[4]).days / 365.2425
            res[item][0] = delta_years
        elif x[8] == None and x[9] != None:
            fim_vig = x[5]
            for i in aux9:
                if i[0] in {'1', '2', '3'}:
                    if (datetime.strptime(i[1], '%Y-%m-%d').date()-x[4]).days < 0:
                        fim_vig = x[4]
                    else:
                        fim_vig = datetime.strptime(i[1], '%Y-%m-%d').date()

            delta_years = 0
            delta_years += relativedelta(fim_vig, x[4]).years
            delta_years += relativedelta(fim_vig, x[4]).months / 12
            delta_years += relativedelta(fim_vig, x[4]).days / 365.2425
            res[item][0] = delta_years
        elif x[8] != None and x[9] == None:
            aux_dict_cas = {}
            aux_dict_rcd = {}
            aux_dict_app = {}
            aux_dict_out = {}
            for i in aux8:
                if i[0] in {'1'}:
                    if float(i[1]) > 1:
                        if i[2] not in aux_dict_cas.keys():
                            aux_dict_cas[i[2]] = float(i[1])
                        else:
                            aux_dict_cas[i[2]] += float(i[1])
                elif i[0] in {'2', '3', '4'}:
                    if float(i[1]) > 1:
                        if i[2] not in aux_dict_rcd.keys():
                            aux_dict_rcd[i[2]] = float(i[1])
                        else:
                            aux_dict_rcd[i[2]] += float(i[1])
                elif i[0] in {'5', '6', '7'}:
                    if float(i[1]) > 1:
                        if i[2] not in aux_dict_app.keys():
                            aux_dict_app[i[2]] = float(i[1])
                        else:
                            aux_dict_app[i[2]] += float(i[1])
                elif i[0] in {'8'}:
                    if float(i[1]) > 1:
                        if i[2] not in aux_dict_out.keys():
                            aux_dict_out[i[2]] = float(i[1])
                        else:
                            aux_dict_out[i[2]] += float(i[1])

            for i in aux_dict_cas.values():
                res[item][83].append(i)
            for i in aux_dict_rcd.values():
                res[item][84].append(i)
            for i in aux_dict_app.values():
                res[item][85].append(i)
            for i in aux_dict_out.values():
                res[item][86].append(i)

            delta_years = 0
            delta_years += relativedelta(x[5], x[4]).years
            delta_years += relativedelta(x[5], x[4]).months / 12
            delta_years += relativedelta(x[5], x[4]).days / 365.2425
            res[item][0] = delta_years
        else:
            fim_vig = x[5]
            for i in aux9:
                if i[0] in {'1', '2', '3'}:
                    if (datetime.strptime(i[1], '%Y-%m-%d').date()-x[4]).days < 0:
                        fim_vig = x[4]
                    else:
                        fim_vig = datetime.strptime(i[1], '%Y-%m-%d').date()
            
            delta_years = 0
            delta_years += relativedelta(fim_vig, x[4]).years
            delta_years += relativedelta(fim_vig, x[4]).months / 12
            delta_years += relativedelta(fim_vig, x[4]).days / 365.2425
            res[item][0] = delta_years
            
            aux_dict_cas = {}
            aux_dict_rcd = {}
            aux_dict_app = {}
            aux_dict_out = {}
            for i in aux8:
                if i[0] in {'1'}:
                    if float(i[1]) > 1 and (fim_vig-datetime.strptime(i[2], '%Y-%m-%d').date()).days > 0:
                        if i[2] not in aux_dict_cas.keys():
                            aux_dict_cas[i[2]] = float(i[1])
                        else:
                            aux_dict_cas[i[2]] += float(i[1])
                elif i[0] in {'2', '3', '4'}:
                    if float(i[1]) > 1 and (fim_vig-datetime.strptime(i[2], '%Y-%m-%d').date()).days > 0:
                        if i[2] not in aux_dict_rcd.keys():
                            aux_dict_rcd[i[2]] = float(i[1])
                        else:
                            aux_dict_rcd[i[2]] += float(i[1])
                elif i[0] in {'5', '6', '7'}:
                    if float(i[1]) > 1 and (fim_vig-datetime.strptime(i[2], '%Y-%m-%d').date()).days > 0:
                        if i[2] not in aux_dict_app.keys():
                            aux_dict_app[i[2]] = float(i[1])
                        else:
                            aux_dict_app[i[2]] += float(i[1])
                elif i[0] in {'8'}:
                    if float(i[1]) > 1 and (fim_vig-datetime.strptime(i[2], '%Y-%m-%d').date()).days > 0:
                        if i[2] not in aux_dict_out.keys():
                            aux_dict_out[i[2]] = float(i[1])
                        else:
                            aux_dict_out[i[2]] += float(i[1])

            for i in aux_dict_cas.values():
                res[item][83].append(i)
            for i in aux_dict_rcd.values():
                res[item][84].append(i)
            for i in aux_dict_app.values():
                res[item][85].append(i)
            for i in aux_dict_out.values():
                res[item][86].append(i)

        if x[0] == '2':
            res[item][65] = 1

        if x[10] == '1':
            res[item][66] = 1
        if x[10] == '2':
            res[item][67] = 1
        if x[10] == '3':
            res[item][68] = 1
        if x[10] == '4':
            res[item][69] = 1
        if x[10] == '5':
            res[item][70] = 1
        if x[10] == '6':
            res[item][71] = 1
        if x[10] == '7':
            res[item][72] = 1
        if x[10] == '8':
            res[item][73] = 1
        if x[10] == '9':
            res[item][74] = 1

        if x[11] == '1':
            res[item][75] = 1
        if x[11] == '3':
            res[item][76] = 1
        if x[11] == '4':
            res[item][77] = 1
        if x[11] == '9':
            res[item][78] = 1

        res[item][79] = x[12]

        res[item][80] = x[13]
        res[item][81] = x[14] + x[15] + x[16]
        res[item][82] = x[17] + x[18] + x[19]

    return res


if __name__ == '__main__':
    months = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')
    years = ('08', '09', '10', '11')
    for mmm in months:
        for aa in years:
            filename = 'data_' + mmm + aa + '_raw.pkl'
            data0 = load_pkl(filename)
            data = data_transf(data0)
            data = [item for item in data if item[0] > 1e-2]
            y_cas = []
            y_rcd = []
            y_app = []
            y_out = []
            X = np.empty([len(data), len(data[0])-4])
            for i, item in enumerate(data):
                y_cas.append(item[-4])
                y_rcd.append(item[-3])
                y_app.append(item[-2])
                y_out.append(item[-1])
                X[i] = item[:-4]
            
            X = np.hstack((np.log(X[:,[0]]), X[:,1:]))
            results = dict([('y_cas', y_cas), ('y_rcd', y_rcd), ('y_app', y_app), ('y_out', y_out), ('X', X)])

            try:
                os.remove('/home/pgsqldata/Susep/data_' + mmm + aa + '.pkl')
            except OSError:
                pass
        
            with open('/home/pgsqldata/Susep/data_' + mmm + aa + '.pkl', 'wb') as file:
                pickle.dump(results, file)

            print('File data_' + mmm + aa + '.pkl saved') 
