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

# IPCA for claim adjustment, below is hardcoded jan-2008 to dez-2016, with jan-2008 as base value 1.0
dates = ['2008-01', '2008-02', '2008-03', '2008-04', '2008-05', '2008-06', '2008-07', '2008-08', '2008-09', '2008-10', '2008-11', '2008-12', '2009-01', '2009-02', '2009-03', '2009-04', '2009-05', '2009-06', '2009-07', '2009-08', '2009-09', '2009-10', '2009-11', '2009-12', '2010-01', '2010-02', '2010-03', '2010-04', '2010-05', '2010-06', '2010-07', '2010-08', '2010-09', '2010-10', '2010-11', '2010-12', '2011-01', '2011-02', '2011-03', '2011-04', '2011-05', '2011-06', '2011-07', '2011-08', '2011-09', '2011-10', '2011-11', '2011-12', '2012-01', '2012-02', '2012-03', '2012-04', '2012-05', '2012-06', '2012-07', '2012-08', '2012-09', '2012-10', '2012-11', '2012-12', '2013-01', '2013-02', '2013-03', '2013-04', '2013-05', '2013-06', '2013-07', '2013-08', '2013-09', '2013-10', '2013-11', '2013-12', '2014-01', '2014-02', '2014-03', '2014-04', '2014-05', '2014-06', '2014-07', '2014-08', '2014-09', '2014-10', '2014-11', '2014-12', '2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06', '2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12', '2016-01', '2016-02', '2016-03', '2016-04', '2016-05', '2016-06', '2016-07', '2016-08', '2016-09', '2016-10', '2016-11', '2016-12']
factors = [1.00000000, 1.00540000, 1.01032646, 1.01517603, 1.02075950, 1.02882350, 1.03643679, 1.04192990, 1.04484731, 1.04756391, 1.05227795, 1.05606615, 1.05902313, 1.06410645, 1.06995903, 1.07209895, 1.07724502, 1.08230808, 1.08620438, 1.08881127, 1.09044449, 1.09306156, 1.09612213, 1.10061623, 1.10468851, 1.11297368, 1.12165487, 1.12748748, 1.13391415, 1.13878999, 1.13878999, 1.13890386, 1.13935943, 1.14448654, 1.15307019, 1.16264067, 1.16996531, 1.17967602, 1.18911343, 1.19850743, 1.20773593, 1.21341229, 1.21523241, 1.21717678, 1.22168034, 1.22815524, 1.23343631, 1.23985018, 1.24604943, 1.25302731, 1.25866593, 1.26130913, 1.26938151, 1.27395128, 1.27497044, 1.28045281, 1.28570267, 1.29303118, 1.30066006, 1.30846402, 1.31880089, 1.33014257, 1.33812343, 1.34441261, 1.35180688, 1.35680856, 1.36033627, 1.36074437, 1.36401015, 1.36878419, 1.37658626, 1.38401983, 1.39675281, 1.40443495, 1.41412555, 1.42713550, 1.43669731, 1.44330612, 1.44907934, 1.44922425, 1.45284731, 1.46112854, 1.46726528, 1.47474834, 1.48625137, 1.50468089, 1.52303800, 1.54314210, 1.55409841, 1.56559874, 1.57796697, 1.58775036, 1.59124341, 1.59983613, 1.61295478, 1.62924563, 1.64488638, 1.66577644, 1.68076843, 1.68799573, 1.69829251, 1.71153919, 1.71752957, 1.72646073, 1.73405716, 1.73544440, 1.73995656, 1.74308848]
cpi = dict(zip(dates, factors))

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
    res[item][87] -> lista, valores indenizados corrigidos casco
    res[item][88] -> lista, valores indenizados corrigidos rcd
    res[item][89] -> lista, valores indenizados corrigidos app
    res[item][90] -> lista, valores indenizados corrigidos outros
    '''

    res = []
    for item, x in enumerate(data):
        res.append([0] * 91)
        res[item][1] = 1
        res[item][83] = []
        res[item][84] = []
        res[item][85] = []
        res[item][86] = []
        res[item][87] = []
        res[item][88] = []
        res[item][89] = []
        res[item][90] = []
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
            aux_dict2_cas = {}
            aux_dict2_rcd = {}
            aux_dict2_app = {}
            aux_dict2_out = {}
            for i in aux8:
                # remove claims where year is bellow inicio_vig:
                if int(i[2][:4]) < int(str(x[4])[:4]):
                    continue

                if i[0] in {'1'}:
                    if float(i[1]) >= 10:
                        if i[2] not in aux_dict_cas.keys():
                            aux_dict_cas[i[2]] = float(i[1])
                            aux_dict2_cas[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                        else:
                            aux_dict_cas[i[2]] += float(i[1])
                            aux_dict2_cas[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                elif i[0] in {'2', '3', '4'}:
                    if float(i[1]) >= 10:
                        if i[2] not in aux_dict_rcd.keys():
                            aux_dict_rcd[i[2]] = float(i[1])
                            aux_dict2_rcd[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                        else:
                            aux_dict_rcd[i[2]] += float(i[1])
                            aux_dict2_rcd[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                elif i[0] in {'5', '6', '7'}:
                    if float(i[1]) >= 10:
                        if i[2] not in aux_dict_app.keys():
                            aux_dict_app[i[2]] = float(i[1])
                            aux_dict2_app[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                        else:
                            aux_dict_app[i[2]] += float(i[1])
                            aux_dict2_app[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                elif i[0] in {'8'}:
                    if float(i[1]) >= 5:
                        if i[2] not in aux_dict_out.keys():
                            aux_dict_out[i[2]] = float(i[1])
                            aux_dict2_out[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                        else:
                            aux_dict_out[i[2]] += float(i[1])
                            aux_dict2_out[i[2]] += float(i[1]) / cpi[i[2][:-3]]

            for i in aux_dict_cas.values():
                res[item][83].append(i)
            for i in aux_dict_rcd.values():
                res[item][84].append(i)
            for i in aux_dict_app.values():
                res[item][85].append(i)
            for i in aux_dict_out.values():
                res[item][86].append(i)
            for i in aux_dict2_cas.values():
                res[item][87].append(i)
            for i in aux_dict2_rcd.values():
                res[item][88].append(i)
            for i in aux_dict2_app.values():
                res[item][89].append(i)
            for i in aux_dict2_out.values():
                res[item][90].append(i)

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
            aux_dict2_cas = {}
            aux_dict2_rcd = {}
            aux_dict2_app = {}
            aux_dict2_out = {}
            for i in aux8:
                # remove claims where year is bellow inicio_vig:
                if int(i[2][:4]) < int(str(x[4])[:4]):
                    continue

                if i[0] in {'1'}:
                    if float(i[1]) >= 10 and (fim_vig-datetime.strptime(i[2], '%Y-%m-%d').date()).days > 0:
                        if i[2] not in aux_dict_cas.keys():
                            aux_dict_cas[i[2]] = float(i[1])
                            aux_dict2_cas[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                        else:
                            aux_dict_cas[i[2]] += float(i[1])
                            aux_dict2_cas[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                elif i[0] in {'2', '3', '4'}:
                    if float(i[1]) >= 10 and (fim_vig-datetime.strptime(i[2], '%Y-%m-%d').date()).days > 0:
                        if i[2] not in aux_dict_rcd.keys():
                            aux_dict_rcd[i[2]] = float(i[1])
                            aux_dict2_rcd[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                        else:
                            aux_dict_rcd[i[2]] += float(i[1])
                            aux_dict2_rcd[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                elif i[0] in {'5', '6', '7'}:
                    if float(i[1]) >= 10 and (fim_vig-datetime.strptime(i[2], '%Y-%m-%d').date()).days > 0:
                        if i[2] not in aux_dict_app.keys():
                            aux_dict_app[i[2]] = float(i[1])
                            aux_dict2_app[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                        else:
                            aux_dict_app[i[2]] += float(i[1])
                            aux_dict2_app[i[2]] += float(i[1]) / cpi[i[2][:-3]]
                elif i[0] in {'8'}:
                    if float(i[1]) >= 5 and (fim_vig-datetime.strptime(i[2], '%Y-%m-%d').date()).days > 0:
                        if i[2] not in aux_dict_out.keys():
                            aux_dict_out[i[2]] = float(i[1])
                            aux_dict2_out[i[2]] = float(i[1]) / cpi[i[2][:-3]]
                        else:
                            aux_dict_out[i[2]] += float(i[1])
                            aux_dict2_out[i[2]] += float(i[1]) / cpi[i[2][:-3]]

            for i in aux_dict_cas.values():
                res[item][83].append(i)
            for i in aux_dict_rcd.values():
                res[item][84].append(i)
            for i in aux_dict_app.values():
                res[item][85].append(i)
            for i in aux_dict_out.values():
                res[item][86].append(i)
            for i in aux_dict2_cas.values():
                res[item][87].append(i)
            for i in aux_dict2_rcd.values():
                res[item][88].append(i)
            for i in aux_dict2_app.values():
                res[item][89].append(i)
            for i in aux_dict2_out.values():
                res[item][90].append(i)

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
            y_cas_cpi = []
            y_rcd_cpi = []
            y_app_cpi = []
            y_out_cpi = []
            X = np.empty([len(data), len(data[0])-8])
            for i, item in enumerate(data):
                y_cas.append(item[-8])
                y_rcd.append(item[-7])
                y_app.append(item[-6])
                y_out.append(item[-5])
                y_cas_cpi.append(item[-4])
                y_rcd_cpi.append(item[-3])
                y_app_cpi.append(item[-2])
                y_out_cpi.append(item[-1])
                X[i] = item[:-8]
            
            X = np.hstack((np.log(X[:,[0]]), X[:,1:]))
            results = dict([('y_cas', y_cas), ('y_rcd', y_rcd), ('y_app', y_app), ('y_out', y_out), ('y_cas_cpi', y_cas_cpi), ('y_rcd_cpi', y_rcd_cpi), ('y_app_cpi', y_app_cpi), ('y_out_cpi', y_out_cpi), ('X', X)])

            try:
                os.remove('/home/pgsqldata/Susep/data_' + mmm + aa + '.pkl')
            except OSError:
                pass
        
            with open('/home/pgsqldata/Susep/data_' + mmm + aa + '.pkl', 'wb') as file:
                pickle.dump(results, file)

            print('File data_' + mmm + aa + '.pkl saved') 
