#########################################
## Data transformation for regressions ##
#########################################

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
    base class: ano_modelo=0, passeio nac, regiao met sp, sexo masc, contrato V.M.R., classe bonus=0, tipo franquia = 2 
    res[item][0] -> dummy dif(ano modelo, ano apolice) = 1
    res[item][1] -> dummy dif(ano modelo, ano apolice) = 2
    res[item][2] -> dummy dif(ano modelo, ano apolice) = 3
    res[item][3] -> dummy dif(ano modelo, ano apolice) = 4
    res[item][4] -> dummy dif(ano modelo, ano apolice) = 5
    res[item][5] -> dummy dif(ano modelo, ano apolice) in [6,10]
    res[item][6] -> dummy dif(ano modelo, ano apolice) in [11,20]
    res[item][7] -> dummy dif(ano modelo, ano apolice) > 20
    res[item][8] -> dummy cod_tarif = 11
    res[item][9] -> dummy cod_tarif = 14A
    res[item][10] -> dummy cod_tarif = 14B
    res[item][11] -> dummy cod_tarif = 14C
    res[item][12] -> dummy cod_tarif = 15
    res[item][13] -> dummy cod_tarif = 16
    res[item][14] -> dummy cod_tarif = 17
    res[item][15] -> dummy cod_tarif = 18
    res[item][16] -> dummy cod_tarif = 19
    res[item][17] -> dummy cod_tarif = 20
    res[item][18] -> dummy cod_tarif = 21
    res[item][19] -> dummy cod_tarif = 22
    res[item][20] -> dummy cod_tarif = 23
    res[item][21] -> dummy regiao = 01
    res[item][22] -> dummy regiao = 02
    res[item][23] -> dummy regiao = 03
    res[item][24] -> dummy regiao = 04
    res[item][25] -> dummy regiao = 05
    res[item][26] -> dummy regiao = 06
    res[item][27] -> dummy regiao = 07
    res[item][28] -> dummy regiao = 08
    res[item][29] -> dummy regiao = 09
    res[item][30] -> dummy regiao = 10
    res[item][31] -> dummy regiao = 12
    res[item][32] -> dummy regiao = 13
    res[item][33] -> dummy regiao = 14
    res[item][34] -> dummy regiao = 15
    res[item][35] -> dummy regiao = 16
    res[item][36] -> dummy regiao = 17
    res[item][37] -> dummy regiao = 18
    res[item][38] -> dummy regiao = 19
    res[item][39] -> dummy regiao = 20
    res[item][40] -> dummy regiao = 21
    res[item][41] -> dummy regiao = 22
    res[item][42] -> dummy regiao = 23
    res[item][43] -> dummy regiao = 24
    res[item][44] -> dummy regiao = 25
    res[item][45] -> dummy regiao = 26
    res[item][46] -> dummy regiao = 27
    res[item][47] -> dummy regiao = 28
    res[item][48] -> dummy regiao = 29
    res[item][49] -> dummy regiao = 30
    res[item][50] -> dummy regiao = 31
    res[item][51] -> dummy regiao = 32
    res[item][52] -> dummy regiao = 33
    res[item][53] -> dummy regiao = 34
    res[item][54] -> dummy regiao = 35
    res[item][55] -> dummy regiao = 36
    res[item][56] -> dummy regiao = 37
    res[item][57] -> dummy regiao = 38
    res[item][58] -> dummy regiao = 39
    res[item][59] -> dummy regiao = 40
    res[item][60] -> dummy regiao = 41
    res[item][61] -> dummy sexo = F
    res[item][62] -> continuous idade
    res[item][63] -> k variable - claims count
    res[item][64] -> d variable - exposure in years
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
    res[item][81] -> continuous is_rcdmat
    res[item][82] -> continuous is_rcdc
    res[item][83] -> continuous is_rcdmor
    res[item][84] -> continuous is_app_ma
    res[item][85] -> continuous is_app_ipa
    res[item][86] -> continuous is_app_dmh
    res[item][87] -> lista, valores indenizados
    '''

    res = []
    for item, x in enumerate(data):
        res.append([0] * 88)
        res[item][87] = []
        if x[1] >= 2000 + int(aa):
            pass
        elif (2000 + int(aa) - x[1]) == 1:
            res[item][0] = 1
        elif (2000 + int(aa) - x[1]) == 2:
            res[item][1] = 1
        elif (2000 + int(aa) - x[1]) == 3:
            res[item][2] = 1
        elif (2000 + int(aa) - x[1]) == 4:
            res[item][3] = 1
        elif (2000 + int(aa) - x[1]) == 5:
            res[item][4] = 1
        elif (2000 + int(aa) - x[1]) >= 6 and (2000 + int(aa) - x[1]) <= 10:
            res[item][5] = 1
        elif (2000 + int(aa) - x[1]) >= 11 and (2000 + int(aa) - x[1]) <= 20:
            res[item][6] = 1
        elif (2000 + int(aa) - x[1]) > 20:
            res[item][7] = 1

        if x[2] == ' 11':
            res[item][8] = 1
        if x[2] == '14A':
            res[item][9] = 1
        if x[2] == '14B':
            res[item][10] = 1
        if x[2] == '14C':
            res[item][11] = 1
        if x[2] == ' 15':
            res[item][12] = 1
        if x[2] == ' 16':
            res[item][13] = 1
        if x[2] == ' 17':
            res[item][14] = 1
        if x[2] == ' 18':
            res[item][15] = 1
        if x[2] == ' 19':
            res[item][16] = 1
        if x[2] == ' 20':
            res[item][17] = 1
        if x[2] == ' 21':
            res[item][18] = 1
        if x[2] == ' 22':
            res[item][19] = 1
        if x[2] == ' 23':
            res[item][20] = 1

        if x[3] == '01':
            res[item][21] = 1
        if x[3] == '02':
            res[item][22] = 1
        if x[3] == '03':
            res[item][23] = 1
        if x[3] == '04':
            res[item][24] = 1
        if x[3] == '05':
            res[item][25] = 1
        if x[3] == '06':
            res[item][26] = 1
        if x[3] == '07':
            res[item][27] = 1
        if x[3] == '08':
            res[item][28] = 1
        if x[3] == '09':
            res[item][29] = 1
        if x[3] == '10':
            res[item][30] = 1
        if x[3] == '12':
            res[item][31] = 1
        if x[3] == '13':
            res[item][32] = 1
        if x[3] == '14':
            res[item][33] = 1
        if x[3] == '15':
            res[item][34] = 1
        if x[3] == '16':
            res[item][35] = 1
        if x[3] == '17':
            res[item][36] = 1
        if x[3] == '18':
            res[item][37] = 1
        if x[3] == '19':
            res[item][38] = 1
        if x[3] == '20':
            res[item][39] = 1
        if x[3] == '21':
            res[item][40] = 1
        if x[3] == '22':
            res[item][41] = 1
        if x[3] == '23':
            res[item][42] = 1
        if x[3] == '24':
            res[item][43] = 1
        if x[3] == '25':
            res[item][44] = 1
        if x[3] == '26':
            res[item][45] = 1
        if x[3] == '27':
            res[item][46] = 1
        if x[3] == '28':
            res[item][47] = 1
        if x[3] == '29':
            res[item][48] = 1
        if x[3] == '30':
            res[item][49] = 1
        if x[3] == '31':
            res[item][50] = 1
        if x[3] == '32':
            res[item][51] = 1
        if x[3] == '33':
            res[item][52] = 1
        if x[3] == '34':
            res[item][53] = 1
        if x[3] == '35':
            res[item][54] = 1
        if x[3] == '36':
            res[item][55] = 1
        if x[3] == '37':
            res[item][56] = 1
        if x[3] == '38':
            res[item][57] = 1
        if x[3] == '39':
            res[item][58] = 1
        if x[3] == '40':
            res[item][59] = 1
        if x[3] == '41':
            res[item][60] = 1

        if x[6] == 'F':
            res[item][61] = 1

        res[item][62] += relativedelta(x[4], x[7]).years
        res[item][62] += relativedelta(x[4], x[7]).months / 12
        res[item][62] += relativedelta(x[4], x[7]).days / 365.2425

        if x[8] == None and x[9] == None:
            res[item][63] = 0
            delta_years = 0
            delta_years += relativedelta(x[5], x[4]).years
            delta_years += relativedelta(x[5], x[4]).months / 12
            delta_years += relativedelta(x[5], x[4]).days / 365.2425
            res[item][64] = delta_years
        elif x[8] == None and x[9] != None:
            fim_vig = x[5]
            for i in x[9].values():
                if i['f1'] in {'2', '3'}:
                    if (datetime.strptime(i['f2'], '%Y-%m-%d').date()-x[4]).days < 0:
                        fim_vig = x[4]
                    else:
                        fim_vig = datetime.strptime(i['f2'], '%Y-%m-%d').date()

            res[item][63] = 0
            delta_years = 0
            delta_years += relativedelta(fim_vig, x[4]).years
            delta_years += relativedelta(fim_vig, x[4]).months / 12
            delta_years += relativedelta(fim_vig, x[4]).days / 365.2425
            res[item][64] = delta_years
        elif x[8] != None and x[9] == None:
            sin_count = 0
            aux_dict = {}
            for j, k in zip(x[8].keys(), x[8].values()):
                if j in {'1', '2', '3', '4', '5', '6'}:
                    if k['f1'] > 0:
                        if k['f2'] not in aux_dict.keys():
                            aux_dict[k['f2']] = k['f1']
                            sin_count += 1
                        else:
                            aux_dict[k['f2']] += k['f1']

            for i in aux_dict.values():
                res[item][87].append(i)

            res[item][63] = sin_count
            delta_years = 0
            delta_years += relativedelta(x[5], x[4]).years
            delta_years += relativedelta(x[5], x[4]).months / 12
            delta_years += relativedelta(x[5], x[4]).days / 365.2425
            res[item][64] = delta_years
        else:
            sin_count = 0
            aux_dict = {}
            fim_vig = x[5]
            for j, k in zip(x[8].keys(), x[8].values()):
                if j in {'1', '2', '3', '4', '5', '6'}:
                    if k['f1'] > 0:
                        if k['f2'] not in aux_dict.keys():
                            aux_dict[k['f2']] = k['f1']
                            sin_count += 1
                        else:
                            aux_dict[k['f2']] += k['f1']

            for i in aux_dict.values():
                res[item][87].append(i)

            for i in x[9].values():
                if i['f1'] in {'2', '3'}:
                    if (datetime.strptime(i['f2'], '%Y-%m-%d').date()-x[4]).days < 0:
                        fim_vig = x[4]
                    else:
                        fim_vig = datetime.strptime(i['f2'], '%Y-%m-%d').date()
            
            res[item][63] = sin_count
            delta_years = 0
            delta_years += relativedelta(fim_vig, x[4]).years
            delta_years += relativedelta(fim_vig, x[4]).months / 12
            delta_years += relativedelta(fim_vig, x[4]).days / 365.2425
            res[item][64] = delta_years

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
        res[item][81] = x[14]
        res[item][82] = x[15]
        res[item][83] = x[16]
        res[item][84] = x[17]
        res[item][85] = x[18]
        res[item][86] = x[19]

    return res


if __name__ == "__main__":
    months = ('jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez')
    years = ('08', '09', '10', '11')
    for mmm in months:
        for aa in years:
            filename = 'data_' + mmm + aa + '_raw.pkl'
            data = load_pkl(filename)
            results = data_transf(data)

            try:
                os.remove('Data/data_' + mmm + aa + '.pkl')
            except OSError:
                pass
        
            with open('Data/data_' + mmm + aa + '.pkl', 'wb') as file:
                pickle.dump(results, file)

            print('File data_' + mmm + aa + '.pkl saved') 
